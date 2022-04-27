import os
import sys
import time


import csv
import json
import random
import pickle
from collections import defaultdict

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix
import pandas as pd

import torch


def read_csv(path, skip_header=False):
    path = path + '.csv' if path[-4:] != '.csv' and path[-4:] != '.dat' and path[-4:] != '.txt' else path
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if skip_header:
            next(reader)
        return [row for row in reader]


def write_csv(path, rows, header=None):
    path = path + '.csv' if path[-4:] != '.csv' else path
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def read_json(path, file=True):
    path = path + '.json' if path[-5:] != '.json' else path
    if not file:
        return json.loads(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_json(path, data, file=True):
    path = path + '.json' if path[-5:] != '.json' else path
    if not file:
        return json.dumps(path, sort_keys=True, indent=4, separators=(',', ': '))
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))


def read_pickle(path):
    path = path + '.pkl' if path[-4:] != '.pkl' else path
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(path, data):
    path = path + '.pkl' if path[-4:] != '.pkl' else path
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_npy(path):
    path = path + '.npy' if path[-4:] != '.npy' else path
    # array = np.load(path)
    with open(path, 'rb') as f:
        array = np.load(f)
    return array


def write_npy(path, array):
    path = path + '.npy' if path[-4:] != '.npy' else path
    # np.save(path, array)
    with open(path, 'wb') as f:
        np.save(f, array)


def read_npz(path):
    path = path + '.npz' if path[-4:] != '.npz' else path
    # sparse_matrix = scipy.sparse.load_npz(path)
    with open(path, 'rb') as f:
        sparse_matrix = scipy.sparse.load_npz(f)
    return sparse_matrix


def write_npz(path, sparse_matrix):
    path = path + '.npz' if path[-4:] != '.npz' else path
    # scipy.sparse.save_npz(path, sparse_matrix)
    with open(path, 'wb') as f:
        scipy.sparse.save_npz(f, sparse_matrix)


def to_array(*args, **kwargs):
    """
    https://pytorch.org/docs/stable/tensors.html#torch.Tensor.numpy
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.column_or_1d.html
    """
    a = []
    for v in args:
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, **kwargs)
        a.append(v)
    return tuple(a)


def to_tensor(*args, **kwargs):
    """

    https://pytorch.org/docs/stable/generated/torch.as_tensor.html
    https://pytorch.org/docs/stable/generated/torch.from_numpy.html
    """
    a = []
    for v in args:
        if not isinstance(v, torch.Tensor):
            v = torch.Tensor(v, **kwargs)
        a.append(v)
    return tuple(a)


def as_tensor(*args, **kwargs):
    a = []
    for v in args:
        v = torch.as_tensor(v, **kwargs)
        a.append(v)
    return tuple(a)


def pause(seconds=None):
    if seconds is None:
        # os.system("pause")
        input("press any key to continue.")
    time.sleep(int(seconds))
    return True


def exit():
    sys.exit()
