#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import sys
import time
import random
import pickle
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import scipy as sp
import pandas as pd
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix

from tqdm import tqdm
from torchtext.vocab import Vocab

from byob.utils import read_json, write_json, read_pickle, write_pickle, read_csv, write_csv

data_dir = "/root/reclib/data"
# data_dir = r"C:\Users\\Desktop\code\reclib\data"
data_dir = os.path.join(data_dir, 'movielens')

csv_file = os.path.join(data_dir, 'ratings.dat')
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(csv_file, sep='::', header=None, names=columns)
print(df.shape, df.columns)
print(df.head())

print("# of ratings: %d" % len(df))
print("# of users: %d" % len(df['user_id'].unique()))
print("# of movies: %d" % len(df['item_id'].unique()))

print(len(df['user_id'].unique()), df['user_id'].min(), df['user_id'].max())
print(len(df['item_id'].unique()), df['item_id'].min(), df['item_id'].max())

# user_set = set(df['user_id'].unique())
# item_set = set(df['item_id'].unique())
# print(len(user_set), min(user_set), max(user_set))
# print(len(item_set), min(item_set), max(item_set))


df = df.sort_values(by=['user_id', 'timestamp'])
# df = df.sort_values(by=['timestamp'])

click = defaultdict(list)
buy = defaultdict(list)
for index, row in df.iterrows():
    # print(index, row, type(row), sep='\n'); break
    click[row['user_id']].append(row['item_id'])
    if row['rating'] == 5:
        buy[row['user_id']].append(row['item_id'])

lens = [len(v) for k, v in click.items()]
print(len(click), min(lens), max(lens))
lens = [len(v) for k, v in buy.items()]
print(len(buy), min(lens), max(lens))


csv_file = os.path.join(data_dir, 'click.csv')
data = [[k, len(v), '|'.join(map(str, v))] for k, v in click.items()]
write_csv(csv_file, data)

csv_file = os.path.join(data_dir, 'buy.csv')
data = [[k, len(v), '|'.join(map(str, v))] for k, v in buy.items()]
write_csv(csv_file, data)

csv_file = os.path.join(data_dir, 'seqs.csv')
data = [[k, len(v), '|'.join(map(str, v)), '|'.join(map(str, buy[k]))] for k, v in click.items()]
write_csv(csv_file, data)


def user_iterator(seqs):
    for row in seqs:
        tokens = [int(row[0])]
        yield tokens


def item_iterator(seqs):
    for row in seqs:
        tokens = list(map(int, row[2].split('|')))
        yield tokens


def build_vocab(iterator, num_lines=None, min_freq=1):
    counter = Counter()
    with tqdm(unit_scale=0, unit='lines', total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    vocab = Vocab(counter, min_freq=min_freq, specials=('<pad>',), specials_first=True)
    return vocab


csv_file = os.path.join(data_dir, 'seqs.csv')
data = read_csv(csv_file)
print(len(data), data[0])

user_vocab = build_vocab(user_iterator(data), min_freq=1)
item_vocab = build_vocab(item_iterator(data), min_freq=1)

# NOTE THAT DOING THIS！
user_set = set([user_vocab[tok] for tok in user_vocab.freqs])
item_set = set([item_vocab[tok] for tok in item_vocab.freqs])

pkl_file = osp.join(data_dir, 'vocab.user.pkl')
write_pickle(pkl_file, user_vocab)
user_vocab = read_pickle(pkl_file)
assert len(user_set) + 1 == len(user_vocab)

pkl_file = osp.join(data_dir, 'vocab.item.pkl')
write_pickle(pkl_file, item_vocab)
item_vocab = read_pickle(pkl_file)
assert len(item_set) + 1 == len(item_vocab)


seqs = []
for user, _, click, buy in data:
    # print(user, _, click, buy)
    user = int(user)
    click = list(map(int, click.split('|')))
    if len(buy) > 0:
        buy = list(map(int, buy.split('|')))
    else:
        buy = []
    user = user_vocab[user]
    click = [item_vocab[tok] for tok in click]
    buy = [item_vocab[tok] for tok in buy]
    seqs.append((user, _, click, buy))

csv_file = os.path.join(data_dir, 'seqs.csv')
data = [[user, len(click), '|'.join(map(str, click)), '|'.join(map(str, buy))] for user, _, click, buy in seqs]
write_csv(csv_file, data)


seq_len = 20
N = 20
K = 3


dataset = []
for i, (user, _, click, buy) in enumerate(seqs):
    if len(buy) <= 0:
        continue
    neg_set = item_set - set(buy)
    for pos in buy:
        # seq = random.sample(click, k=seq_len)
        idx = np.random.randint(0, len(click) - seq_len + 1)
        seq = click[idx:idx + seq_len]
        neg = random.sample(neg_set, k=1)[0]
        dataset.append((user, seq, pos, 1))
        dataset.append((user, seq, neg, 0))
    if i > 0 and i % 1000 == 0:
        print(i)
print(len(dataset), dataset[0])

csv_file = os.path.join(data_dir, 'train_item_%d_%d.csv' % (N, K))
dataset = [[u, i, y, '|'.join(map(str, seq))] for u, seq, i, y in dataset]
write_csv(csv_file, dataset)


dataset = []
for i, (user, _, click, buy) in enumerate(seqs):
    if len(buy) <= 0:
        continue
    neg_set = item_set - set(buy)
    for pos in buy:
        # seq = random.sample(click, k=seq_len)
        idx = np.random.randint(0, len(click) - seq_len + 1)
        seq = click[idx:idx + seq_len]
        neg = random.sample(neg_set, k=N-1)
        cand = [pos] + neg  # candidate items
        random.shuffle(cand)
        # (user, historical behaviors, positive bundle, candidate items)
        dataset.append((user, seq, pos, cand))  
    if i > 0 and i % 1000 == 0:
        print(i)
print(len(dataset), dataset[0])

csv_file = os.path.join(data_dir, 'train_item_rank_%d_%d.csv' % (N, K))
dataset = [[u, i, '|'.join(map(str, cand)), '|'.join(map(str, seq))] for u, seq, i, cand in dataset]
write_csv(csv_file, dataset)


dataset = []
for i, (user, _, click, buy) in enumerate(seqs):
    if len(buy) <= 0:
        continue
    if len(buy) < K:
        continue
    neg_set = item_set - set(buy)
    # n = int(np.sqrt(len(buy)))
    for _ in range(len(buy)):
        # seq = random.sample(click, k=seq_len)
        idx = np.random.randint(0, len(click) - seq_len + 1)
        seq = click[idx:idx + seq_len]
        # pos = random.sample(buy, k=top_k)
        # pos = np.random.choice(buy, size=top_k, replace=False)
        idx = np.random.randint(0, len(buy) - K + 1)
        pos = buy[idx:idx + K]
        random.shuffle(pos)
        # (user, historical behaviors, positive bundle, candidate items)
        neg = random.sample(neg_set, k=K)
        random.shuffle(neg)
        dataset.append((user, seq, pos, 1))
        dataset.append((user, seq, neg, 0))
    if i > 0 and i % 1000 == 0:
        print(i)
print(len(dataset), dataset[0])

csv_file = os.path.join(data_dir, 'train_bundle_%d_%d.csv' % (N, K))
dataset = [[u, '|'.join(map(str, b)), y, '|'.join(map(str, seq))] for u, seq, b, y in dataset]
write_csv(csv_file, dataset)


dataset = []
for i, (user, _, click, buy) in enumerate(seqs):
    if len(buy) <= 0:
        continue
    if len(buy) < K:
        continue
    neg_set = item_set - set(buy)
    # n = int(np.sqrt(len(buy)))
    for _ in range(len(buy)):
        # seq = random.sample(click, k=seq_len)
        idx = np.random.randint(0, len(click) - seq_len + 1)
        seq = click[idx:idx + seq_len]
        # pos = random.sample(buy, k=top_k)
        # pos = np.random.choice(buy, size=top_k, replace=False)
        idx = np.random.randint(0, len(buy) - K + 1)
        pos = buy[idx:idx + K]
        random.shuffle(pos)
        # (user, historical behaviors, positive bundle, candidate items)
        neg = random.sample(neg_set, k=N-K)
        cand = pos + neg  # candidate items
        random.shuffle(cand)
        dataset.append((user, seq, pos, cand))
    if i > 0 and i % 1000 == 0:
        print(i)
print(len(dataset), dataset[0])

csv_file = os.path.join(data_dir, 'train_bundle_rank_%d_%d.csv' % (N, K))
dataset = [[u, '|'.join(map(str, b)), '|'.join(map(str, cand)), '|'.join(map(str, seq))] for u, seq, b, cand in dataset]
write_csv(csv_file, dataset)


dataset = []
for i, (user, _, click, buy) in enumerate(seqs):
    # user identity
    # -----------------------------------------------------------------
    if len(buy) == 0:
        continue
    if len(buy) < K:
        continue
    # historical behaviors
    # -----------------------------------------------------------------
    # seq = random.sample(click, k=seq_len)
    idx = np.random.randint(0, len(click) - seq_len + 1)
    seq = click[idx:idx + seq_len]
    # positive bundle
    # -----------------------------------------------------------------
    # pos = random.sample(buy, k=K)
    # pos = np.random.choice(buy, size=K, replace=False)
    idx = np.random.randint(0, len(buy) - K + 1)
    pos = buy[idx:idx + K]
    random.shuffle(pos)
    # candidate items
    # -----------------------------------------------------------------
    neg_set = item_set - set(buy)
    neg = random.sample(neg_set, k=N-K)
    cand = pos + neg  # candidate items
    random.shuffle(cand)
    # (user identity, historical behaviors, candidate items, positive bundle)
    dataset.append((user, seq, pos, cand))
    if i > 0 and i % 1000 == 0:
        print(i)
print(len(dataset), dataset[0])

csv_file = os.path.join(data_dir, 'test_bundle_%d_%d.csv' % (N, K))
dataset = [[u, '|'.join(map(str, b)), '|'.join(map(str, cand)), '|'.join(map(str, seq))] for u, seq, b, cand in dataset]
write_csv(csv_file, dataset)
