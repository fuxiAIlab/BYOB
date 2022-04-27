import os
import csv
import json
import time
import random
import pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix

from byob.config import DATA_CONFIG as data_conf
from byob.utils import read_csv, read_pickle, read_npz, write_npz

data_dir = "/project/"
data_set = "yoochoose"
seq_file = data_conf[data_set]['seq_file']
min_freq = data_conf[data_set]['min_freq']
win_size = data_conf[data_set]['win_size']

csv_file = os.path.join(data_dir, data_set, seq_file)
data = read_csv(csv_file)

pkl_file = os.path.join(data_dir, data_set, 'vocab-%d.pkl' % min_freq)
vocab = read_pickle(pkl_file)
print('Vocab has {} entries'.format(len(vocab)))
print({'<unk>': vocab['<unk>'], '<pad>': vocab['<pad>']}, vocab[vocab.UNK], vocab.unk_index)


def get_context_pair(a, win_size=5, vocab=None, include_unk=True):
    if not include_unk:
        a = [v for v in a if vocab[v] != vocab.unk_index]
    pairs = []
    for i in range(len(a)):
        for j in range(max(0, i - win_size // 2), min(len(a) - 1, i + win_size // 2) + 1):
            if j != i:
                if vocab is None:
                    pairs.append((a[i], a[j]))
                else:
                    pairs.append((vocab[a[i]], vocab[a[j]]))
    return pairs


def build_cooccurrence_matrix(data):
    mat = dok_matrix((len(vocab), len(vocab)), dtype=np.int8)
    for i, row in enumerate(data):
        t = get_context_pair(row[2].split(';'), win_size, vocab)
        for a, b in t:
            mat[a, b] = mat[a, b] + 1
        if i > 0 and i % 100000 == 0:
            print(i)
    print(mat.shape, mat.nnz)
    print(type(mat), mat.shape, mat.ndim, mat.dtype, mat.size)
    return mat


def build_cooccurrence_matrix_(data):
    values = []
    for i, row in enumerate(data):
        t = get_context_pair(row[2].split(';'), win_size)
        values.extend(t)
        if i > 0 and i % 1000000 == 0:
            print(i)
    print(len(values), values[:3])
    values = [(vocab[v[0]], vocab[v[1]]) for v in values]
    values = list(zip(*values))
    row = values[0]
    col = values[1]
    data = [1] * len(row)
    mat = csr_matrix((data, (row, col)), shape=(len(vocab), len(vocab)), dtype=np.int8)
    print(mat.shape, mat.nnz)
    print(type(mat), mat.shape, mat.ndim, mat.dtype, mat.size)
    return mat


start_time = time.time()
mat = build_cooccurrence_matrix(data)
# mat = build_cooccurrence_matrix_(data)
elapsed = int(time.time() - start_time)
print(elapsed)
npz_file = os.path.join(data_dir, data_set, '%s-freq-%d-win-%d.npz' % (data_set, min_freq, win_size))
# mat = read_npz(npz_file).todok()
write_npz(npz_file, mat.tocoo())
