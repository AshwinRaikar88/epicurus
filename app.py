import os.path
import cv2

from glob import glob
from text_detector import TextDetector
from text_recognition import TextRecognition


def check_intersection(m2, refpoint=(0, 10)):
    if refpoint[0] < m2 < refpoint[1]:
        return True
    return False


if __name__ == '__main__':
    td = TextDetector()
    tr = TextRecognition()
    # image_path = r"/home/sladmin/Downloads/OCR_images/download.png"
    image_path = r"/home/sladmin/Downloads/OCR_images/1.jpeg"

    img = cv2.imread(image_path)
    output = img.copy()
    H, W, _ = img.shape

    result = td.detect(image_path, conf=0.5, offset=5)

    print("Bounding Boxes from TD predictions")
    print(result)

    # Matrix array sorting
    line_count = 0
    line_array = []
    line_data = {}

    r_y1 = 0
    r_y2 = 0
    int_count = 0
    for i in sorted(result.keys()):
        x1, y1, x2, y2 = result[i]
        # roi = img[y1:y2, x1:x2]
        m1 = int((x2+x1)/2)
        m2 = int((y2+y1)/2)
        cv2.circle(img, (m1, m2), 4, (0, 0, 255), 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.line(img, (0, int((y1+y2)/2)), (W, int((y1+y2)/2)), (255, 0, 0), 2)

        if int_count > 0:
            if check_intersection(m2, (r_y1, r_y2)):
                cv2.line(img, (0, int((y1 + y2) / 2)), (W, int((y1 + y2) / 2)), (128, 128, 128), 2)
                line_flag = True
                line_array.append(result[i])
            else:
                int_count = 0
                line_count += 1
                line_data.update({line_count: line_array})
                line_array = [result[i]]

        else:
            line_array.append(result[i])

        r_y1 = y1
        r_y2 = y2
        int_count += 1

        # cv2.imshow("crop", img)
        # cv2.waitKey(0)

    print("Line Data array")
    print(line_data)

    for line_num in sorted(line_data.keys()):
        print(f"Line {line_num}")
        if not os.path.exists(f"detections/line_{line_num}"):
            os.mkdir(f"detections/line_{line_num}")

        for line in sorted(line_data[line_num]):
            print(line)
            x1, y1, x2, y2 = line
            roi = output[y1:y2, x1:x2]
            cv2.imwrite(f'detections/line_{line_num}/{x1}.png', roi)

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.imshow("crop", output)
            cv2.waitKey(0)

    #
    # images = []
    # for image in glob(r"./detections/*.png"):
    #     images.append(image)
    #
    # tr.predict(images)

