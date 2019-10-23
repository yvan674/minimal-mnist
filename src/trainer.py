"""Trainer

Performs training on the network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms.transforms import ToTensor

from .model import FCNetwork


class Trainer:
    def __init__(self, root: str):
        """Creates the trainer class.

        Args:
            root: path to the MNIST data root.
        """
        self.root = root
        self.train_data = MNIST(root, transform=ToTensor())
        self.test_data = MNIST(root, train=False, transform=ToTensor())

    def train(self, batch_size: int, epochs):
        """Performs training on the network.
        Args:
            optimizer: Either "Adam" or "SGD"
            optimizer_args: The arguments for the specified optimizer.
            batch_size: Batch size for training and validation
            leaky: Use leaky ReLUs or not.
        """
        train_loader = DataLoader(self.train_data, batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size, shuffle=True)
        network = FCNetwork(784, 10, (True, True, False))
        optimizer = SGD(network.parameters(), 0.0001, 1.0)
        loss_crit = CrossEntropyLoss()

        prev_epoch_val_acc = 0

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                img, cls = data
                h1, h2, out = network(img)
                out = out.softmax(1)
                loss = loss_crit(out, data[1])

                # Do backprop
                if i % 100 == 0:
                    print("Iteration {},    \tepoch: {}, \tLoss: {},  "
                          "\taccuracy: {}"
                          .format(i + 1, epoch + 1, loss.item(),
                                  self.calc_batch_accuracy(out, cls)))
                loss.backward()
                optimizer.step()

            # Do validation
            validation_acc = 0
            for i, data in enumerate(test_loader):
                img, cls = data
                h1, h2, out = network(img)
                out = out.softmax(1)

                validation_acc += self.calc_batch_accuracy(out, cls)
            validation_acc /= i
            print("Epoch {} validation accuracy: {}".format(epoch,
                                                            validation_acc))

            if prev_epoch_val_acc - validation_acc > 0.1:
                print("Overfitted to the training set.")
                break

    def calc_batch_accuracy(self, output: torch.Tensor,
                            target: torch.tensor)-> float:
        """Calculates accuracy for a batch.

        Args:
            output: Output predictions of the network on one-hot encoding.
            target: Targets for the predictions.
        """
        oa = output.argmax(1)  # output argmax
        correct = (oa == target).sum()

        return float(correct) / float(target.shape[0])


if __name__ == '__main__':
    t = Trainer('/Users/Yvan/Documents/Projects/Physical NN/MNIST')
    t.train(128, 30)
