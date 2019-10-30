"""Inference.

Run only inference using the network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.transforms import ToTensor

from time import sleep

from model import FCNetwork


class Inference:
    def __init__(self, data_root: str, weights_path: str):
        """Creates the inference class.

        Args:
            data_root: Path to the MNIST data root.
            weights_path: Path to the weights.
        """
        self.root = data_root
        self.test_data = MNIST(data_root, train=False, transform=ToTensor())
        self.weights_path = weights_path

    def run(self):
        """Performs the actual inference."""
        test_loader = DataLoader(self.test_data, 1, shuffle=True)

        # Load the checkpoint
        checkpoint = torch.load(self.weights_path)
        msd = checkpoint['model_state_dict']

        input_value = msd['fc0.0.weight'].shape[1]
        layer1 = msd['fc0.0.weight'].shape[0]
        layer2 = msd['fc1.0.weight'].shape[0]
        classes = msd['fc2.weight'].shape[0]

        network = FCNetwork(input_value, classes, layer1, layer2,
                            (False, True, False))
        network.load_state_dict(msd)

        network.eval()

        for i, data in enumerate(test_loader):
            img, cls = data
            h1, h2, out = network(img)
            out = out.softmax(1)
            out = out.argmax(1)

            print("Predicted value: {} \t Actual value: {} \t Correct: {}"
                  .format(out.item(), cls.item(), out.item() == cls.item()))
            sleep(0.5)


if __name__ == '__main__':
    i = Inference('/workspace/MNIST/', '/workspace/best-model-16.pth')
    i.run()
