"""
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]
"""

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from config.config import common_config as config
from dataset.dataset import Synth90kDataset, synth90k_collate_fn
from models.model import CRNN
from models.ctc_decoder import ctc_decode


class TextRecognition:
    def __init__(self):
        print("[INFO] loading TR model...")

        reload_checkpoint = r"./weights/text_recognition.pt"
        self.decode_method = 'beam_search'
        self.beam_size = 10
        self.batch_size = 1

        self.img_height = config['img_height']
        self.img_width = config['img_width']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f'device: {device}')

        num_class = len(Synth90kDataset.LABEL2CHAR) + 1
        self.crnn = CRNN(1, self.img_height, self.img_width, num_class,
                    map_to_seq_hidden=config['map_to_seq_hidden'],
                    rnn_hidden=config['rnn_hidden'],
                    leaky_relu=config['leaky_relu'])
        self.crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
        self.crnn.to(device)
        print("[INFO] TD model successfully loaded")

    def predict(self, images):
        predict_dataset = Synth90kDataset(paths=images,
                                          img_height=self.img_height, img_width=self.img_width)

        predict_loader = DataLoader(
            dataset=predict_dataset,
            batch_size=self.batch_size,
            shuffle=False)

        preds = self.get_preds(self.crnn, predict_loader, Synth90kDataset.LABEL2CHAR, decode_method=self.decode_method,
                beam_size=self.beam_size)

        self.show_result(images, preds)

    @staticmethod
    def get_preds(crnn, dataloader, label2char, decode_method, beam_size):
        crnn.eval()
        pbar = tqdm(total=len(dataloader), desc="Predict")

        all_preds = []
        with torch.no_grad():
            for data in dataloader:
                device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

                images = data.to(device)

                logits = crnn(images)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                                   label2char=label2char)
                all_preds += preds

                pbar.update(1)
            pbar.close()

        return all_preds

    @staticmethod
    def show_result(paths, preds):
        print('\n===== result =====')

        for path, pred in zip(paths, preds):
            text = ''.join(pred)
            print(f'{path} > {text}')


if __name__ == '__main__':
    from glob import glob
    ep_tr = TextRecognition()

    images = []
    for image in glob(r"./detections/*.png"):
        images.append(image)

    ep_tr.predict(images)


