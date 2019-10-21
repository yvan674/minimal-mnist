"""Search Worker.

Searches for the optimal hyperparameters using HpBandSter, which is an
implementation of BOHB from the AutoML group at Uni Freiburg.

The hyperparameters we will be optimizing are as follows:

+----------------+----------------+------------------+------------------------+
| Parameter name | Parameter type |      Range       |        Comment         |
+----------------+----------------+------------------+------------------------+
| Learning rate  | float          | [1e-5, 1e-2]     | varied logarithmically |
| Optimizer      | categorical    | {'adam', 'sgd'}  | choose one             |
| SGD momentum   | float          | [0, 0.99]        | only active when       |
|                |                |                  | optimizer == 'sgd'     |
| Adam epsilon   | float          | [1e-2, 1]        | only active when       |
|                |                |                  | optimizer == 'adam'    |
| Batch size     | int            | [4, 256]         |                        |
| First layer    | int            | [16, 64]         | Number of nodes in the |
|                |                |                  | first FCN layer        |
| Second layer   | int            | [8, 64]         | Number of nodes in the |
|                |                |                  | second FCN layer       |
| Leaky ReLU     | categorical    | {bool, bool,     | The set of boolean     |
|                |                | bool}            | combinations           |
+----------------+----------------+------------------+------------------------+

+-------------------+----------------+
|  Parameter name   | Name in config |
+-------------------+----------------+
| Learning Rate     | lr             |
| Optimizer         | optimizer      |
| SGD momentum      | momentum       |
| Adam epsilon      | epsilon        |
| Batch size        | bs             |
| First layer       | first_layer    |
| Second layer      | second_layer   |
| Leaky ReLU        | leaky          |
+-------------------+----------------+

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        <http://proceedings.mlr.press/v80/falkner18a.html>
"""
# Torch imports
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import FCNetwork

import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SearchWorker(Worker):
    def __init__(self, data_path, logging_path, **kwargs):
        """Initializes the search worker.

        Args:
            data_path (str): Path to the data directory.
            iaa: The image augmentation module imported. This is necessary
                because sometimes, importing it here doesn't work but importing
                it in a different module does.
            logging_dir (str): Path to the logging directory. Used for logging
                configuration, loss, accuracy. Ideally, this is a subdirectory
                from the output directory.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.run_count = 0

        self.logging_path = logging_path
        self.train_data = MNIST(data_path, transform=ToTensor())
        self.test_data = MNIST(data_path, train=False, transform=ToTensor())


    def compute(self, config, budget, **kwargs):
        """Runs the training session.

        This training session will also save all the data on its runs (e.g.
        config, loss, accuracy) into the logging dir

        Args:
            config (dict): Dictionary containing the configuration by the
                optimizer
            budget (int): Amount of epochs the model can use to train.

        Returns:
            dict: dictionary with fields 'loss' (float) and 'info' (dict)
        """
        leaky_configs = (
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [False, True, True],
            [True, False, False],
            [True, False, True],
            [True, True, False],
            [True, True, True]
        )

        # Start with printouts
        print("\nStarting run {} with config:.".format(self.run_count))
        print("    Optimizer: {}".format(config['optimizer']))
        print("    Learning rate: {}".format(config['lr']))
        print("    Batch size: {}".format(config['bs']))
        print("    First layer: {}".format(config['first_layer']))
        print("    Second layer: {}".format(config['second_layer']))
        print("    Leaky config: {}".format(leaky_configs[config['leaky']]))
        # Set network, dataloader, optimizer, and loss criterion
        train_loader = DataLoader(self.train_data, config['bs'], shuffle=True)
        test_loader = DataLoader(self.test_data, config['bs'], shuffle=True)

        network = FCNetwork(784, 10, config['first_layer'],
                            config['second_layer'],
                            leaky_configs[config['leaky']])

        if config['optimizer'] == 'sgd':
            optimizer = SGD(network.parameters(), config['lr'],
                            config['momentum'])
        else:
            optimizer = Adam(network.parameters(), config['lr'],
                             config['epsilon'])
        loss_crit = CrossEntropyLoss()

        # Increment run count number
        self.run_count += 1

        # Start actual training loop
        for epoch in range(int(budget)):

            # Do training loop
            network.train()
            for i, (img, cls) in enumerate(train_loader):
                optimizer.zero_grad()
                h1, h2, out = network(img)
                out = out.softmax(1)
                loss = loss_crit(out, cls)

                # Do backprop
                if i % int(1000 / (config['bs'] / 4)) == 0:
                    print("Iteration {},    \tepoch: {}, \tLoss: {},  "
                          "\taccuracy: {}"
                          .format(i + 1, epoch + 1, loss.item(),
                                  self.calc_batch_accuracy(out, cls)))
                loss.backward()
                optimizer.step()

        train_loss, train_acc = self.evaluate_network(
            network, loss_crit, train_loader
        )
        validation_loss, validation_accuracy = self.evaluate_network(
            network, loss_crit, test_loader)

        return {'loss': 1 - validation_accuracy,
                'info': {'validation accuracy': validation_accuracy,
                         'validation loss': validation_loss,
                         'training loss': train_loss,
                         'training accuracy': train_acc
                         }
                }

    def evaluate_network(self, network, criterion, data_loader):
        """Evaluate network accuracy on a specific data set.

        Returns:
            list: Element-wise accuracy
            float: Average loss
        """
        # Set to eval and set up variables
        network.eval()
        loss_val = 0
        acc_val = 0

        # Use network but without updating anything
        with torch.no_grad():
            for i, (img, cls) in enumerate(data_loader):
                h1, h2, out = network(img)
                out = out.softmax(1)
                loss_val += criterion(out, cls).item()

                correct = (out.argmax(1) == cls).sum()

                acc_val += float(correct) / float(cls.shape[0])

        # Average accuracy and loss
        loss_val /= i
        acc_val /= i
        return loss_val, acc_val

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

    @staticmethod
    def get_configspace():
        """Builds the config space as described in the header docstring."""
        cs = CS.ConfigurationSpace()

        lr = CS.UniformFloatHyperparameter('lr',
                                           lower=1e-5,
                                           upper=1e-2,
                                           default_value=1e-4,
                                           log=True)

        optimizer = CS.CategoricalHyperparameter('optimizer', ['adam', 'sgd'])
        momentum = CS.UniformFloatHyperparameter('momentum', lower=0.,
                                                 upper=1.00,
                                                 default_value=0.9)
        epsilon = CS.UniformFloatHyperparameter('epsilon', lower=1e-2,
                                                upper=1.,
                                                default_value=0.1)
        bs = CS.UniformIntegerHyperparameter('bs', lower=4, upper=256)

        first_layer = CS.UniformIntegerHyperparameter('first_layer', lower=16,
                                                      upper = 64)
        second_layer = CS.UniformIntegerHyperparameter('second_layer', lower=8,
                                                       upper=64)

        leaky = CS.CategoricalHyperparameter('leaky', [0, 1, 2, 3, 4, 5,
                                                       6, 7])
        cs.add_hyperparameters([lr, optimizer, momentum, epsilon, bs,
                                first_layer, second_layer, leaky])
        cs.add_condition(CS.EqualsCondition(momentum, optimizer, 'sgd'))
        cs.add_condition(CS.EqualsCondition(epsilon, optimizer, 'adam'))

        return cs
