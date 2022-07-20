import time
import cv2
import numpy as np

from scipy.interpolate import UnivariateSpline
from imutils.object_detection import non_max_suppression


class TextDetector:
    def __init__(self, Debug=False):
        self.DEBUG = Debug
        self.weights_path = 'weights/text_detection.pb'
        self.min_conf = 0.9
        self.width = 1024   # resized image width (should be multiple of 32)
        self.height = 1024  # resized image height (should be multiple of 32)

        # Define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading TD model...")
        self.net = cv2.dnn.readNet(self.weights_path)
        print("[INFO] TD model successfully loaded")

    def detect(self, path, conf=0.9, offset=10):
        output_dict = {}
        self.min_conf = conf

        image = cv2.imread(path)
        orig = image.copy()
        if self.DEBUG:
            debug_img = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (self.width, self.height)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        # c = 255 / np.log(1 + np.max(image))
        # log_image = c * (np.log(image + 1))
        # log_image = np.array(log_image, dtype = np.uint8)
        # image2 = warm_img(image)
        (H, W) = image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layerNames)
        end = time.time()
        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < self.min_conf:
                    continue
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # Loop over the bounding boxes
        bbcount = 0

        for (startX, startY, endX, endY) in boxes:
            # Scale the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            bbcount += 1
            # output_dict.update({bbcount: (startX, startY, endX, endY)})
            r1 = startY
            r2 = endY

            s_key = ((r1 + r2)/2 * startY) / W

            s_key *= s_key
            output_dict.update({int(s_key): (startX, startY, endX, endY)})
            roi = orig[startY-offset:endY+offset, startX-offset:endX+offset]

            if self.DEBUG:
                # Draw the bounding box on the image
                cv2.rectangle(debug_img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.imshow('crop.jpg', roi)
                cv2.waitKey(0)

            # cv2.imwrite(f'detections/{bbcount}.png', roi)

        # Show the output image
        if self.DEBUG:
            cv2.imshow("Text Detection", debug_img)
            cv2.waitKey(0)

        return output_dict

    @staticmethod
    def create_LUT_8UC1(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    @staticmethod
    def warm_img(img):
        incr_ch_lut = TextDetector.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
        decr_ch_lut = TextDetector.create_LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
        c_b, c_g, c_r = cv2.split(img)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.merge((c_b, c_g, c_r))
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm, cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
        return img_bgr_warm
