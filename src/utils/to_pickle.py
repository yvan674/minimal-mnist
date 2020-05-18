"""To Pickle.

Converts the pytorch tensor MNIST dataset to numpy versions through pickle.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 10, 2020
"""
import pickle
import torch
from argparse import ArgumentParser
from os.path import join


def parse_args():
    p = ArgumentParser(description='converts a pytorch MNIST dataset to a numpy'
                                   'version')
    p.add_argument('ROOT', type=str,
                   help='root of the MNIST directory, which contains the '
                        '``processed`` directory')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    processed_dir = join(args.ROOT, 'processed')

    t = torch.load(join(processed_dir, 'training.pt'))
    n = tuple([i.numpy() for i in t])
    with open(join(processed_dir, 'training.pkl'), 'wb') as fp:
        pickle.dump(n, fp)

    t = torch.load(join(processed_dir, 'test.pt'))
    n = tuple([i.numpy() for i in t])
    with open(join(processed_dir, 'test.pkl'), 'wb') as fp:
        pickle.dump(n, fp)
