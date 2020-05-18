"""To Numpy.

Converts a torch state_dict to a numpy state_dict array

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    April 10, 2020
"""
import torch
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from os.path import join, split, splitext


def parse_args():
    p = ArgumentParser(description='converts a torch state_dict to a numpy '
                                   'state_dict array')
    p.add_argument('FILE')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    state_dict = torch.load(args.FILE)

    np_state_dict = OrderedDict()
    for k, v in state_dict.items():
        np_state_dict[k] = v.detach().cpu().numpy()

    parent_dir, file_name = split(args.FILE)
    file_name = splitext(file_name)[0]
    out_path = join(parent_dir, file_name + '.npy')

    np.save(out_path, np_state_dict)
