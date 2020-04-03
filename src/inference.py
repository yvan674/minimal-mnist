"""Inference.

Runs inference using a trained network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 3, 2020
"""
import torch
from torchvision.datasets import MNIST

import numpy as np

from model import FCNetwork

class AI:
    def __init__(self, root, state_dict_path):
        """Initializes the AI.

        Args:
            root (str): Path to the MNIST data root.
            state_dict_path (str): Path to the weight .pth file
        """
        self.root = root
        self.data = MNIST(root, train=False)

        self.layer_1_neurons = 16
        self.layer_2_neurons = 36

        self.model = FCNetwork(784, 10, self.layer_1_neurons,
                               self.layer_2_neurons, (False, False))

        state_dict = torch.load(state_dict_path)['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
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
            self.counter += 1

        tensor_image = torch.tensor(image / 255, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            h1, h2, out = self.model(tensor_image)
            out = out.argmax(1)

        return image, int(out[0])


if __name__ == '__main__':
    ai = AI('E:\Offline Docs\Git\minimal-mnist\MNIST',
            'E:\Offline Docs\Git\minimal-mnist\\best-model.pth')
    ai.infer_next()
