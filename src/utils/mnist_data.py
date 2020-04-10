"""MNIST Data.

Custom MNIST Dataset utility since torch can't easily be installed on the Pi.
This is basically a copy of the torchvision version, but without the torch
components. Also has significantly reduced functionality, as it is only meant to
retrieve images from the pickle file.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 10, 2020
"""
import os
import pickle
from PIL import Image


class MNIST:
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
         "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
         "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
         "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pkl'
    test_file = 'test.pkl'

    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True):
        """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

        Args:
            root (str): Root directory of dataset where
                ``MNIST/processed/training.pt`` and  ``MNIST/processed/test.pt``
                exist.
            train (bool): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
        """
        self.root = root
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        with open(os.path.join(self.processed_folder, data_file), 'rb') as fp:
            self.data, self.targets = pickle.load(fp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")