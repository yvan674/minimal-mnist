"""Numpy Model.

The model as a series of numpy operations.
"""
import numpy as np


class Linear:
    def __init__(self, in_connections, out_connections):
        """A simple linear layer."""
        self.weight = np.zeros([out_connections, in_connections])
        self.bias = np.zeros([out_connections])
        self.in_connections = in_connections
        self.out_connections = out_connections

    def __call__(self, x):
        """Calculates a linear function.

        Args:
            x (np.ndarray): input.
        """
        # x = np.stack([x.reshape(self.in_connections)] * self.out_connections)
        return np.dot(x, self.weight.T) + self.bias

    def load_state_dict(self, key, value):
        self.__setattr__(key[0], value)


class ReLU:
    def __init__(self):
        """Creates a ReLU activation function."""
        pass

    def __call__(self, x):
        """Calculates using the ReLU activation function.

        Args:
            x (np.ndarray): input.
        """
        return x.clip(min=0)


class Sequential:
    def __init__(self, layers):
        """Creates a sequential function.

        Args:
            layers (list): List of layers to run through sequentially.
        """
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def load_state_dict(self, key, value):
        params = key.split('.')
        self.layers[int(params[0])].load_state_dict(params[1:], value)


class NumpyModel:
    def __init__(self, in_connections: int, num_classes: int, first_layer: int,
                 second_layer: int):
        """Creates the network as a series of Numpy operations."""
        self.in_connections = in_connections

        self.fc0 = Sequential([Linear(in_connections, first_layer),
                               ReLU()])
        self.fc1 = Sequential([Linear(first_layer, second_layer),
                               ReLU()])
        self.fc2 = Linear(second_layer, num_classes)

    def __call__(self, x):
        """Runs the input through the network.

        Args:
            x (np.ndarray): input.
        """
        x = x.reshape([x.shape[0], self.in_connections])
        x1 = self.fc0(x)
        x2 = self.fc1(x1)
        x3 = self.fc2(x2)

        return x1, x2, x3

    def load_state_dict(self, state_dict):
        """Loads the state dictionary"""
        for k, v in state_dict.items():
            params = k.split('.')
            if len(params[1:]) > 1:
                to_get = '.'.join(params[1:])
            else:
                to_get = [params[1]]
            self.__getattribute__(params[0]).load_state_dict(to_get, v)
