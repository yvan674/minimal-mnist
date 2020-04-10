"""Inference.

Runs inference using a trained network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 3, 2020
"""
try:
    import torch
    from torchvision.datasets import MNIST
    from model import FCNetwork as NNModel
    USE_NUMPY = False
except ImportError:
    from model import NumpyModel as NNModel
    from utils.mnist_data import MNIST
    USE_NUMPY = True

import numpy as np
from argparse import ArgumentParser
from time import time


def parse_args():
    parser = ArgumentParser(description='runs inference on the network')
    parser.add_argument('ROOT', type=str,
                        help='path to the root of the dataset')
    parser.add_argument('MODEL', type=str,
                        help='model state_dict to be loaded')
    return parser.parse_args()


class AI:
    def __init__(self, root, state_dict_path):
        """Initializes the AI.

        Args:
            root (str): Path to the MNIST data root.
            state_dict_path (str): Path to the weight .pth file
        """
        self.root = root
        self.data = MNIST(root, train=False)

        if USE_NUMPY:
            state_dict = np.load(state_dict_path, allow_pickle=True).item()
        else:
            state_dict = torch.load(state_dict_path)

        in_connections = state_dict['fc0.0.weight'].shape[1]
        out_connections = state_dict['fc2.bias'].shape[0]

        self.layer_1_neurons = state_dict['fc0.0.bias'].shape[0]
        self.layer_2_neurons = state_dict['fc1.0.bias'].shape[0]

        self.model = NNModel(in_connections,
                             out_connections,
                             self.layer_1_neurons,
                             self.layer_2_neurons)
        if not USE_NUMPY:
            self.model.eval()

        self.model.load_state_dict(state_dict)
        self.counter = 0

    def infer_next(self, image=None) -> (np.ndarray, int):
        """Infer the next number in the list or the given image.

        Args:
            image (np.ndarray): A 28 x 28 array representing the image. Should
                be converted directly from PIL.Image to np.ndarray using
                np.array(img). This has max value 255 and min value 0.

        Returns:
            The input image and the prediction.
        """
        if image is None:
            if self.counter == len(self.data):
                # Reset counter if it reaches the end.
                self.counter = 0

            image = np.array(self.data[self.counter][0])

        if USE_NUMPY:
            image_arr = image.reshape([1, 1, image.shape[0], image.shape[1]])
            image_arr = image_arr.astype(dtype=float) / 255.
            h1, h2, out = self.model(image_arr)
            out = out.argmax(1)
        else:
            tensor_image = torch.tensor(image, dtype=torch.float) / 255.
            tensor_image = tensor_image.unsqueeze(0)

            with torch.no_grad():
                h1, h2, out = self.model(tensor_image)
                out = out.argmax(1)

        self.counter += 1
        return image, int(out[0]), h1, h2


if __name__ == '__main__':
    args = parse_args()
    print(f"Running on {'numpy' if USE_NUMPY else 'torch'}")

    print('loading model...')
    start_time = time()
    ai = AI(args.ROOT, args.MODEL)
    print(f"done! t={time() - start_time:.3f}s")

    start_time = time()
    for i in range(10000):
        _, out, _, _ = ai.infer_next()
        if i % 1000 == 0:
            print(f'Iteration {i}: pred={out}')
    time_del = time() - start_time
    time_del /= 10000

    print(f'done! t per iter={time_del:.6f}s')

