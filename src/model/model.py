"""Model.

The neural-network model.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import torch.nn as nn


class FCNetwork(nn.Module):
    def __init__(self, in_connections: int, num_classes: int,
                 first_layer: int, second_layer: int,
                 leaky: tuple = (False, False)):
        """Creates the FC network.

        Args:
            in_connections: Number of incoming connections
            num_classes: Number of final classes.
            leaky: Each item in the list corresponds to the leaky value of an
                fc layer.
        """
        super(FCNetwork, self).__init__()
        self.in_connections = in_connections
        relus = (nn.ReLU() if not leaky[0] else nn.LeakyReLU(),
                 nn.ReLU() if not leaky[1] else nn.LeakyReLU())
        self.fc0 = nn.Sequential(nn.Linear(in_connections, first_layer),
                                 relus[0])
        self.fc1 = nn.Sequential(nn.Linear(first_layer, second_layer),
                                 relus[1])
        self.fc2 = nn.Linear(second_layer, num_classes)

    def forward(self, x) -> tuple:
        """Runs forward on the network.

        Args:
            x (torch.Tensor): Input as a [n, 1, 28, 28] or [n, 28, 28]
                shaped tensor.
        """
        x = x.reshape(x.shape[0], self.in_connections)
        x1 = self.fc0(x)
        x2 = self.fc1(x1)
        x3 = self.fc2(x2)

        # We return the result of every layer for visualization purposes
        return x1, x2, x3
