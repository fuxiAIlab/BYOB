import os
import os.path as osp
import sys
import csv
import argparse
import traceback
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from byob.config import data_dir, model_dir, output_dir, DATA_CONFIG, MODEL_CONFIG, DEFAULT_CONFIG
from byob.data_utils import SequenceDataset, setup_dataset_item, setup_dataset_bundle, \
    setup_dataset_item_bpr, setup_dataset_bundle_bpr, setup_dataset_test, setup_dataset_test_v1
# from byob.pipeline import Pipeline
from byob.utils import read_json, write_json, read_pickle, write_pickle, read_csv, write_csv
from byob.metrics import binary_accuracy, categorical_accuracy, bundle_metrics

from byob.models.baseline_ncf import ItemNCFModel, BundleNCFModel
from byob.models.baseline_bpr import ItemBPRModel, BundleBPRModel
from byob.models.baseline_rnn import ItemRNNModel
from byob.models.baseline_cnn import ItemCNNModel
from byob.models.baseline_trm import ItemTRMModel

conf = dict()
conf['time'] = str(datetime.now())
conf['date'] = str(datetime.today().date())
conf['torch'] = torch.__version__
conf['cuda'] = '%s (%s)' % (torch.cuda.is_available(), torch.version.cuda)
conf['data_dir'] = data_dir

parser = argparse.ArgumentParser()
parser.add_argument('--num_seeds', type=int, default=1, help='number of experiment seeds (default: 1)')
parser.add_argument('--ml_task', type=str, default='BIN', choices=('BIN', 'MUL', 'CLS', 'REG'))
parser.add_argument('--dataset', type=str, default='movielens', choices=('movielens', 'yoochoose'))
parser.add_argument('--label', type=str, default='item', choices=('item', 'bundle'))
parser.add_argument('--pool_size', type=int, default=20, help='pool size (default: 20)')
parser.add_argument('--bundle_size', type=int, default=3, help='bundle size (default: 3)')
parser.add_argument('--model', type=str, default='NCF', choices=('BPR', 'NCF', 'RNN', 'CNN', 'TRM'))
parser.add_argument('--model_list', type=str, default='NCF')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-05, help='l2 regularization (default: 1e-05)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

cmd_conf = vars(args)
cmd_conf['model_list'] = cmd_conf['model_list'].split(',')
conf.update(cmd_conf)

# dataset
# --------------------------------------------------------------------------------------------------------------------

# conf['label'] = 'bundle'
conf.update(DATA_CONFIG[conf['dataset']])
# conf['embed_path'] = osp.join(model_dir, '%s-%s.npy' % (conf['dataset'], 'SkipGramModel'))
# conf.update(MODEL_CONFIG[conf['model']])

# conf['device'] = "cpu"
conf['device'] = "cuda" if torch.cuda.is_available() else "cpu"
# if conf['device'] == "cpu":
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pkl_file = osp.join(data_dir, conf['dataset'], conf['user_vocab'])
print("load user vocab:", pkl_file)
user_vocab = read_pickle(pkl_file)
assert conf['n_user'] == len(user_vocab)
pkl_file = osp.join(data_dir, conf['dataset'], conf['item_vocab'])
print("load item vocab:", pkl_file)
item_vocab = read_pickle(pkl_file)
assert conf['n_item'] == len(item_vocab)
conf['vocab_size'] = len(item_vocab)

# print(conf)
for k, v in conf.items():
    print(f'{k} -> {v}')
file_name = 'config-%s-%s.json' % (conf['dataset'], conf['model'])
json_file = osp.join(output_dir, file_name)
write_json(json_file, conf)

if conf['label'] == 'item':
    if conf['model'] in ['NCF', 'RNN', 'CNN', 'TRM']:
        file_name = 'train_item_%d_%d.csv' % (conf['pool_size'], conf['bundle_size'])
        csv_file = osp.join(data_dir, conf['dataset'], file_name)
        dataset = setup_dataset_item(csv_file, item_vocab)
    else:
        csv_file = osp.join(conf['data_dir'], conf['dataset'], conf['seq_file'])
        dataset = setup_dataset_item_bpr(csv_file, item_vocab, conf['seq_len'])
else:
    if conf['model'] in ['NCF', 'RNN', 'CNN', 'TRM']:
        file_name = 'train_bundle_%d_%d.csv' % (conf['pool_size'], conf['bundle_size'])
        csv_file = osp.join(data_dir, conf['dataset'], file_name)
        dataset = setup_dataset_bundle(csv_file, item_vocab)
    else:
        csv_file = osp.join(conf['data_dir'], conf['dataset'], conf['seq_file'])
        dataset = setup_dataset_bundle_bpr(csv_file, item_vocab, conf['seq_len'], conf['bundle_size'])
print(len(dataset), dataset[0])

file_name = 'test_bundle_%d_%d.csv' % (conf['pool_size'], conf['bundle_size'])
csv_file = osp.join(conf['data_dir'], conf['dataset'], file_name)
test_ds = setup_dataset_test(csv_file, item_vocab)
# csv_file = osp.join(conf['data_dir'], conf['dataset'], conf['seq_file'])
# test_ds = setup_dataset_test_v1(csv_file, user_vocab, vocab, conf['seq_len'], conf['pool_size'], conf['bundle_size'])
print(len(test_ds), test_ds[0])

train_len = int(len(dataset) * 0.8)
train_ds, valid_ds = random_split(dataset, [train_len, len(dataset) - train_len])
print(len(dataset), len(train_ds), len(valid_ds))

# vocab = dataset.vocab
# conf['vocab_size'] = len(dataset.vocab)
# # conf['vocab'] = vocab
# print('Vocab has {} entries'.format(len(vocab)))
# print({'<unk>': vocab['<unk>'], '<pad>': vocab['<pad>']})
# pkl_file = os.path.join(data_dir, conf['dataset'], 'vocab-%s.pkl' % str(conf['min_freq']))
# vocab = read_pickle(pkl_file)
# write_pickle(pkl_file, vocab)

# model
# --------------------------------------------------------------------------------------------------------------------


def build_pipeline(name, conf):
    assert conf['label'] in ['item', 'bundle']
    prefix = 'Item' if conf['label'] == 'item' else 'Bundle'
    assert name in MODEL_CONFIG.keys()
    conf.update(MODEL_CONFIG[name])
    model = eval(prefix + name + 'Model')(conf)
    if name == 'BPR':
        conf['logits'] = True
        from byob.pipeline import PipelineBPRSeq as Pipeline
    else:
        conf['logits'] = False
        from byob.pipeline import PipelineUIYSeq as Pipeline
    pl = Pipeline(conf, model.to(conf['device']))
    conf['logits'] = False
    if conf['ml_task'] == "BIN":
        if conf['logits']:
            # loss_fn = nn.BCEWithLogitsLoss()
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            loss_fn = nn.BCELoss()
    elif conf['ml_task'] == "MUL":
        loss_fn = nn.CrossEntropyLoss()
    elif conf['ml_task'] == "REG":
        loss_fn = nn.MSELoss()
    else:
        # loss_fn = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss()
    if name == 'BPR':
        def loss_fn(p_ui, p_uj, mean=False):
            if mean:
                loss = -F.logsigmoid(p_ui - p_uj).mean()
            else:
                loss = -F.logsigmoid(p_ui - p_uj).sum()
                # loss = -(p_ui - p_uj).sigmoid().log().sum()
            return loss
    if conf['logits']:
        metrics = {'accuracy': lambda y_true, y_pred: binary_accuracy(y_true, y_pred, threshold=0.5, sigmoid=True)}
    else:
        metrics = {'accuracy': lambda y_true, y_pred: binary_accuracy(y_true, y_pred, threshold=0.5, sigmoid=False)}
    pl.compile(loss_fn=loss_fn, metrics=metrics)
    return pl


def bundle_predict(pl, dataset, K=3):
    y_true = []
    y_pred = []
    for u, b, cand, seq in dataset:
        y_true.append(b)
        test_ds = []
        for i in cand:
            test_ds.append((u, i, seq))
        test_ds = SequenceDataset(test_ds)
        y_hat = pl.predict(test_ds, len(test_ds))
        # print(y_hat.shape, y_hat.dtype, y_hat[0])
        ind = np.argsort(-y_hat.reshape(-1), axis=0)
        y_pred.append(cand[ind][:K])
    y_true = np.stack(y_true, axis=0)
    y_pred = np.stack(y_pred, axis=0)
    return y_true, y_pred


def bundle_predict_seq(pl, dataset, K):
    x, y = dataset
    test_ds = SequenceDataset(x)
    y_pred = []
    for i in range(K):
        y_hat = pl.predict(test_ds)
        # print(y_hat.shape, y_hat.dtype, y_hat[0])
        y_hat = y_hat.reshape(-1, 1)
        y_pred.append(y_hat)
        x = np.concatenate([x[:, 1:], y_hat], axis=1)
        test_ds = SequenceDataset(x)
    y_pred = np.concatenate(y_pred, axis=1)
    # print(y_pred.shape, y_pred.dtype, y_pred[0])
    return y, y_pred


# fit
# --------------------------------------------------------------------------------------------------------------------

# pl = Pipeline(conf, model)
# pl.compile(metrics={'accuracy': categorical_accuracy})
# try:
#     history = pl.fit(train_ds, valid_ds, conf['batch_size'], conf['num_epochs'])
#     print(history)
# except Exception as e:
#     print(e)
#     print(traceback.format_exc())

for name in conf['model_list']:
    conf['model'] = name
    history_list = []
    for seed in range(conf['num_seeds']):
        print('-' * 40, "model: %s, seed: %s" % (name, seed), '-' * 40)
        # random.seed(seed)
        np.random.seed(seed)  # reproducibility
        torch.manual_seed(seed)
        # tf.set_random_seed(seed)
        conf['seed'] = seed
        pl = build_pipeline(name, conf)
        # a = pl.model.get_embedding()
        # print(a.shape, a[:3])
        try:
            if conf['label'] == 'item':
                metrics = bundle_metrics(*bundle_predict(pl, test_ds, conf['bundle_size']))
                print("epoch %d metrics: %s" % (0, metrics))
            epoch_history = []
            for epoch in range(1, conf['num_epochs'] + 1):
                hist = pl.fit(train_ds, valid_ds, conf['batch_size'], 1)
                if conf['label'] == 'item':
                    metrics = bundle_metrics(*bundle_predict(pl, test_ds, conf['bundle_size']))
                    print("epoch %d metrics: %s" % (epoch, metrics))
                    hist.update(metrics)
                epoch_history.append(hist)
                # a = pl.model.get_embedding()
                # print(a.shape, a[:3])
            print(epoch_history)
            history_list.append(epoch_history)
            # file_name = 'history-%s-%s-%d.json' % (conf['dataset'], name, seed)
            # json_file = osp.join(output_dir, file_name)
            # write_json(json_file, hist)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
    file_name = 'history-%s-%s-%s-%s.json' % (conf['dataset'], conf['pool_size'], conf['bundle_size'], name)
    json_file = osp.join(output_dir, file_name)
    write_json(json_file, history_list)
