"""Weights and Biases.

Uses a Weights and Biases sweep to find the optimal hyperparameters which
minimizes number of parameters while maximizing the accuracy
"""
import wandb

import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms.transforms import ToTensor

from datetime import datetime as dt
from os.path import join, exists
from pathlib import Path
from argparse import ArgumentParser

from model import FCNetwork

hyperparameter_defaults = {
    'batch_size': 16,
    'first_layer': 10,
    'second_layer': 10,
    'lr': 1e-4,
    'momentum': 0.9,
    'decay': 0.1,
    'epochs': 250,
}

wandb.init(config=hyperparameter_defaults, project="fully_connected_mnist")


def parse_args():
    """Parses arguments from the command line."""
    p = ArgumentParser(description="trainer for use with weights and biases")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--first_layer", type=int)
    p.add_argument("--second_layer", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--momentum", type=float)
    p.add_argument("--decay", type=float)
    p.add_argument("--epochs", type=int)
    return p.parse_args()


def calc_batch_accuracy(output: torch.Tensor,
                        target: torch.Tensor) -> float:
    """Calculates accuracy for a batch.

    Args:
        output: Output predictions of the network on one-hot encoding.
        target: Targets for the predictions.
    """
    oa = output.argmax(1)  # output argmax
    correct = (oa == target).sum()

    return float(correct) / float(target.shape[0])


def train(root: str, results_dir: str, batch_size: int, first_layer: int,
          second_layer: int, lr: float, momentum: float, decay: float,
          epochs: int):
    """Performs training on the network.

    Args:
        root: path to the MNIST data root.
        results_dir: Logging path to use.
        batch_size: Batch size for training and validation.
        first_layer: Number of nodes in the first layer.
        second_layer: Number of nodes in the second layer.
        lr: Learning rate for the optimizer
        momentum: Momentum of the optimizer
        decay: Decay of the optimizer
        epochs: Number of epochs to train for.
    """
    epochs = 150 if epochs is None else epochs

    # Create results directory first
    r_dir = join(results_dir, dt.now().strftime('%y-%m-%d__%H-%M-%S'))
    while exists(r_dir):
        # To make sure that the directory doesn't exist. If it does, try
        # again.
        r_dir = join(results_dir, dt.now().strftime('%y-%m-%d__%H-%M-%S'))
    Path(r_dir).mkdir(parents=True, exist_ok=True)

    # Then load data in
    train_data = MNIST(root, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_data = MNIST(root, train=False, transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    steps_per_epoch = len(train_loader)

    # Load model
    model = FCNetwork(784, 10, first_layer, second_layer, (False, False))

    layer_crit = (first_layer + second_layer) / 80

    # Optimizer and loss function
    optimizer = SGD(model.parameters(), lr, momentum, weight_decay=decay)
    loss_criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        steps_done = steps_per_epoch * epoch
        for i, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            img, cls = data
            _, _, out = model(img)
            loss = loss_criterion(out, cls)

            # Print outs at certain intervals
            if i % 100 == 0:
                print("Iteration {:<8}epoch: {:<3} Loss: {:<2.4f}, "
                      "accuracy: {:.4f}".format(str(i + 1) + ":",
                                                str(epoch + 1) + ",",
                                                loss.item(),
                                                calc_batch_accuracy(out, cls)))
            # Send metrics
            wandb.log({'loss': loss.item()}, step=steps_done + i + 1)
            # Do backprop
            loss.backward()
            optimizer.step()

        # At the end of the epoch, do validation
        validation_acc = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.eval()
                img, cls = data
                _, _, out = model(img)
                validation_acc += calc_batch_accuracy(out, cls)
            validation_acc /= len(test_loader)
            print("Epoch{} validation accuracy: {}".format(epoch,
                                                           validation_acc))
        wandb.log({'accuracy': validation_acc,
                   'maximization_criterion': validation_acc - layer_crit},
                  step=steps_per_epoch * (epoch + 1))  # + 1 since end of epoch

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            join(r_dir, 'epoch_{}.pth'.format(epoch))
        )


if __name__ == '__main__':
    args = parse_args()
    root = 'MNIST/'
    results_dir = 'workdir/'
    train(root, results_dir, **vars(args))
