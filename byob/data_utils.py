import io
import os
import os.path as osp
import random
from collections import Counter, defaultdict
import logging
import time
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
# from torchtext.datasets import text_classification

from tqdm import tqdm

from byob.utils import read_csv, write_csv


class EmbeddingDataset(Dataset):
    r"""Defines an word embedding dataset.
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, data, vocab=None):
        super(EmbeddingDataset, self).__init__()
        self.data = data  # a list of sample tuple
        self.vocab = vocab  # vocabulary object

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x


class SequenceDataset(Dataset):
    r"""Defines an sequence dataset.
    """

    def __init__(self, data, vocab=None):
        super(SequenceDataset, self).__init__()
        self.data = data  # a list of sample tuple
        self.vocab = vocab  # Vocabulary object

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x


def _csv_iterator(csv_path, ngrams=1):
    with io.open(csv_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter=',')
        for row in reader:
            tokens = row[2].split('|')
            # tokens = list(map(int, row[2].split('|')))
            if ngrams > 1:
                yield ngrams_iterator(tokens, ngrams)
            else:
                yield tokens


def _build_vocab(iterator, num_lines=None, min_freq=1):
    counter = Counter()
    with tqdm(unit_scale=0, unit='lines', total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    # freqs = sorted(counter.items(), key=lambda tup: tup[0])
    # idx2tok = [tok for tok in freqs.keys()]
    # tok2idx = {tok: i for i, tok in enumerate(idx2tok)}
    vocab = Vocab(counter, min_freq=min_freq, specials=('<pad>',), specials_first=True)
    return vocab


def _build_proba(vocab, min_freq=1):
    counter = vocab.freqs
    freqs = sorted(counter.items(), key=lambda tup: tup[0])
    token = np.array([vocab[tok] for tok, cnt in freqs])
    freqs = np.array([cnt for tok, cnt in freqs], dtype=np.float32)
    proba = freqs / np.sum(freqs)  # normalization
    proba = proba ** (3. / 4.)  # raise to the 3/4rd power
    proba = proba / np.sum(proba)  # normalization
    # return dict(zip(token, proba.tolist()))
    return token, proba


def setup_dataset_vec(csv_path, vocab, c=5, k=15):
    """
        - c: the size of the context words
        - k: the number of negative samples
    """
    if vocab is None:
        vocab = _build_vocab(_csv_iterator(csv_path), min_freq=1)
    token, proba = _build_proba(vocab)
    print(token.min(), token.max(), proba.min(), proba.max())
    start_time = time.time()
    dataset = []
    generator = _csv_iterator(csv_path)
    for seq in generator:
        # print(seq, list(map(int, seq)), [vocab[tok] for tok in seq], sep='\n'); break
        seq = list(map(int, seq))
        # seq = [vocab[tok] for tok in seq]
        for i in range(c, len(seq) - c):
            for j in range(max(i - c, 0), min(i + c, len(seq))):
                if i == j:
                    continue
                inp = seq[i]  # the input word
                out = seq[j]  # target word in the context
                neg = np.random.choice(token, size=(k,), replace=True, p=proba).tolist()
                dataset.append([inp, out, np.array(neg, dtype=np.int64)])
                # dataset.append([inp, out, torch.tensor(neg, dtype=torch.long)])
    elapsed = time.time() - start_time
    print(len(dataset), dataset[0], elapsed)
    return EmbeddingDataset(dataset, vocab=vocab)


def setup_dataset_item(csv_path, vocab):
    data = read_csv(csv_path)
    # print(len(data), data[0])
    dataset = []
    for u, i, y, seq in data:
        seq = list(map(int, seq.split('|')))
        dataset.append([int(u), int(i), int(y), torch.tensor(seq)])
    # print(len(dataset), dataset[0])
    return SequenceDataset(dataset, vocab=vocab)


def setup_dataset_bundle(csv_path, vocab):
    data = read_csv(csv_path)
    # print(len(data), data[0])
    dataset = []
    for u, b, y, seq in data:
        b = list(map(int, b.split('|')))
        seq = list(map(int, seq.split('|')))
        dataset.append([int(u), torch.tensor(b), int(y), torch.tensor(seq)])
    # print(len(dataset), dataset[0])
    return SequenceDataset(dataset, vocab=vocab)


def setup_dataset_item_bpr(csv_path, vocab, seq_len):
    data = read_csv(csv_path)
    # print(len(data), data[0])
    item_set = set([vocab[tok] for tok in vocab.freqs])
    # print(len(item_set), item_set)
    dataset = []
    for user, _, click, buy in data:
        if buy == '' or len(buy) == 0:
            continue
        user = int(user)
        click = list(map(int, click.split('|')))
        buy = list(map(int, buy.split('|')))
        neg_set = item_set - set(buy)
        for pos in buy:
            # seq = random.sample(click, k=seq_len)
            idx = np.random.randint(0, len(click) - seq_len + 1)
            seq = click[idx:idx + seq_len]
            neg = random.sample(neg_set, k=1)[0]
            # print(user, click, buy, pos, neg, seq)
            dataset.append((user, pos, neg, np.array(seq, dtype=np.int64)))
    # print(len(dataset), dataset[0])
    return SequenceDataset(dataset, vocab=vocab)


def setup_dataset_bundle_bpr(csv_path, vocab, seq_len, K):
    data = read_csv(csv_path)
    # print(len(data), data[0])
    item_set = set([vocab[tok] for tok in vocab.freqs])
    # print(len(item_set), item_set)
    dataset = []
    for user, _, click, buy in data:
        if buy == '' or len(buy) == 0:
            continue
        user = int(user)
        click = list(map(int, click.split('|')))
        buy = list(map(int, buy.split('|')))
        if len(buy) < K:
            continue
        neg_set = item_set - set(buy)
        for _ in buy:
            # seq = random.sample(click, k=seq_len)
            idx = np.random.randint(0, len(click) - seq_len + 1)
            seq = click[idx:idx + seq_len]
            idx = np.random.randint(0, len(buy) - K + 1)
            pos = buy[idx:idx + K]
            neg = random.sample(neg_set, k=K)
            # print(user, click, buy, pos, neg, seq)
            dataset.append((user, np.array(pos, dtype=np.int64), np.array(neg, dtype=np.int64), np.array(seq, dtype=np.int64)))
    # print(len(dataset), dataset[0])
    return SequenceDataset(dataset, vocab=vocab)


def setup_dataset_test(csv_path, vocab=None):
    dataset = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for user, pos, cand, seq in reader:
            pos = list(map(int, pos.split('|')))
            cand = list(map(int, cand.split('|')))
            seq = list(map(int, seq.split('|')))
            dataset.append((int(user), np.array(pos, dtype=np.int64), np.array(cand, dtype=np.int64), np.array(seq, dtype=np.int64)))
    # print(len(dataset), dataset[0])
    return dataset


def setup_dataset_test_v1(csv_path, vocab, seq_len, N, K):
    data = read_csv(csv_path)
    # print(len(data), data[0])
    item_set = set([vocab[tok] for tok in vocab.freqs])
    # print(len(item_set), item_set)
    dataset = []
    for user, _, click, buy in data:
        if buy == '' or len(buy) == 0:
            continue
        # user identity
        # -----------------------------------------------------------------
        user = int(user)
        click = list(map(int, click.split('|')))
        buy = list(map(int, buy.split('|')))
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
        # dataset.append((user, seq, cand, pos))
        dataset.append((user, np.array(pos, dtype=np.int64), np.array(cand, dtype=np.int64), np.array(seq, dtype=np.int64)))
    # print(len(dataset), dataset[0])
    return dataset


def setup_dataset_test_v2(conf, vocab, K):
    feature, label = {}, {}
    csv_file = os.path.join(conf['data_dir'], conf['dataset'], conf['seq_file'])
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            feature[row[0]] = [vocab[v] for v in row[2].split('|')]
            # print(feature); break
    csv_file = os.path.join(conf['data_dir'], conf['dataset'], conf['buy_file'])
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if int(row[1]) < K:
                continue
            label[row[0]] = [vocab[v] for v in row[2].split('|')]
            # print(label); break
    # print(len(feature), len(label))
    dataset = []
    for u in feature:
        if u not in label:
            continue
        # print(u, feature[u], label[u])
        # x = np.ones(conf['max_len'], dtype=np.int64) * vocab['<pad>']
        # x[-len(feature[u]):] = feature[u][-conf['max_len']:]
        # y = np.array(label[u][-conf['top_k']:]).astype(np.int64)
        x = np.random.choice(feature[u], size=(conf['max_len'],), replace=True).astype(np.int64)
        y = np.random.choice(label[u], size=(conf['bundle_size'],), replace=True).astype(np.int64)
        # print(x, x.shape, x.dtype, y, y.shape, y.dtype)
        dataset.append((u, x, y))
    # print(len(dataset), dataset[0])
    return dataset


def gather_movielens_info(vocab=None, ngrams=1, min_freq=1):
    csv_path = os.path.join("/root/reclib/data", 'movielens', 'click.csv')
    if vocab is None:
        vocab = _build_vocab(_csv_iterator(csv_path, ngrams), min_freq=min_freq)
    iterator = _csv_iterator(csv_path, ngrams)
    num_users, num_clicks = 0, 0
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            tokens = list(filter(lambda v: v is not Vocab.UNK, [vocab[token] for token in tokens]))
            num_users += 1
            num_clicks += len(tokens)
            t.update(1)
    csv_path = os.path.join("/root/reclib/data", 'movielens', 'buy.csv')
    iterator = _csv_iterator(csv_path, ngrams)
    num_buys = 0
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            tokens = list(filter(lambda v: v is not Vocab.UNK, [vocab[token] for token in tokens]))
            num_buys += len(tokens)
            t.update(1)
    return {
        '#users': num_users,
        '#items': len(vocab),
        '#clicks': num_clicks,
        '#buys': num_buys,
        '#behaviors': num_clicks + num_buys,
        '#clicks/user': num_clicks / num_users,
        '#buys/user': num_buys / num_users,
        '#behaviors/user': (num_clicks + num_buys) / num_users,
        '#clicks/item': num_clicks / len(vocab),
        '#buys/item': num_buys / len(vocab),
        '#behaviors/item': (num_clicks + num_buys) / len(vocab),
    }


def gather_yoochoose_info(vocab=None, ngrams=1, min_freq=1):
    csv_path = os.path.join("/root/reclib/data", 'yoochoose', 'click.csv')
    if vocab is None:
        vocab = _build_vocab(_csv_iterator(csv_path, ngrams), min_freq=min_freq)
    iterator = _csv_iterator(csv_path, ngrams)
    num_users, num_clicks = 0, 0
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            tokens = list(filter(lambda v: v is not Vocab.UNK, [vocab[token] for token in tokens]))
            num_users += 1
            num_clicks += len(tokens)
            t.update(1)
    csv_path = os.path.join("/root/reclib/data", 'yoochoose', 'buy.csv')
    iterator = _csv_iterator(csv_path, ngrams)
    num_buys = 0
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            tokens = list(filter(lambda v: v is not Vocab.UNK, [vocab[token] for token in tokens]))
            num_buys += len(tokens)
            t.update(1)
    return {
        '#users': num_users,
        '#items': len(vocab),
        '#clicks': num_clicks,
        '#buys': num_buys,
        '#behaviors': num_clicks + num_buys,
        '#clicks/user': num_clicks / num_users,
        '#buys/user': num_buys / num_users,
        '#behaviors/user': (num_clicks + num_buys) / num_users,
        '#clicks/item': num_clicks / len(vocab),
        '#buys/item': num_buys / len(vocab),
        '#behaviors/item': (num_clicks + num_buys) / len(vocab),
    }


def gather_taobao_info(vocab=None, ngrams=1, min_freq=1):
    csv_path = os.path.join("/project/", 'taobao', 'click.csv')
    if vocab is None:
        vocab = _build_vocab(_csv_iterator(csv_path, ngrams), min_freq=min_freq)
    iterator = _csv_iterator(csv_path, ngrams)
    num_users, num_clicks = 0, 0
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            tokens = list(filter(lambda v: v is not Vocab.UNK, [vocab[token] for token in tokens]))
            num_users += 1
            num_clicks += len(tokens)
            t.update(1)
    csv_path = os.path.join("/project/", 'taobao', 'buy.csv')
    iterator = _csv_iterator(csv_path, ngrams)
    num_buys = 0
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            tokens = list(filter(lambda v: v is not Vocab.UNK, [vocab[token] for token in tokens]))
            num_buys += len(tokens)
            t.update(1)
    return {
        '#users': num_users,
        '#items': len(vocab),
        '#clicks': num_clicks,
        '#buys': num_buys,
        '#behaviors': num_clicks + num_buys,
        '#clicks/user': num_clicks / num_users,
        '#buys/user': num_buys / num_users,
        '#behaviors/user': (num_clicks + num_buys) / num_users,
        '#clicks/item': num_clicks / len(vocab),
        '#buys/item': num_buys / len(vocab),
        '#behaviors/item': (num_clicks + num_buys) / len(vocab),
    }


if __name__ == "__main__":

    from byob.config import data_dir
    from seqrec.data_utils import read_pickle, write_pickle

    dataset = 'yoochoose'  # ('movielens', 'yoochoose', 'taobao')

    csv_file = os.path.join(data_dir, dataset, 'seqs.csv')
    vocab = _build_vocab(_csv_iterator(csv_file), min_freq=1)

    pkl_file = os.path.join(data_dir, dataset, 'vocab.user.pkl')
    write_pickle(pkl_file, vocab)

    token, proba = _build_proba(vocab)
    print(token.min(), token.max(), proba.min(), proba.max())
    # print(vocab.freqs, vocab.itos, vocab.stoi, token, proba, sep='\n')

    item_set = [vocab[tok] for tok in vocab.freqs]
    print(len(item_set), min(item_set), max(item_set))

    # setup_dataset(csv_file, c=5, k=15)

    # start_time = time.time()
    # vocab, proba = _build_vocab(_csv_iterator(csv_file), min_freq=1)
    # elapsed = time.time() - start_time
    # print(len(vocab), vocab[0], elapsed)

    # pkl_file = os.path.join(data_dir, 'movielens', 'dataset.pkl')
    # start_time = time.time()
    # # write_pickle(pkl_file, dataset)
    # dataset = read_pickle(pkl_file)
    # elapsed = time.time() - start_time
    # print(len(dataset), dataset[0], elapsed)

    # info = gather_movielens_info()
    # info = gather_yoochoose_info()
    # info = gather_taobao_info()
    # print(info)
