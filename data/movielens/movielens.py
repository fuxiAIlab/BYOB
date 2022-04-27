import os
import time
import pickle

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import dok_matrix

from byob.utils import df2mat, sample_negative, sample_negative_df

# data_dir = "/root/reclib/data"
data_dir = "C:/Users/Desktop/code/reclib/data"
data_dir = os.path.join(data_dir, 'movielens')

csv_file = os.path.join(data_dir, 'ratings.dat')
h5_file = os.path.join(data_dir, 'ratings.h5')
columns = ['user_id', 'item_id', 'rating', 'ts']
df = pd.read_csv(csv_file, sep='::', header=None, names=columns)
# df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')
# df = pd.read_csv(csv_file, sep=',', header='infer')
print(df.shape, df.columns)
print(df.head())

if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

print("# of ratings: %d" % len(df))
print("# of users: %d" % len(df['user_id'].unique()))
print("# of movies: %d" % len(df['item_id'].unique()))


def build_vocab(df, start_idx=0):
    vocab = {}

    role_ids = sorted(df['user_id'].unique())
    n_user = len(role_ids)
    user2id = dict(zip(role_ids, range(start_idx, start_idx + len(role_ids))))
    id2user = {user2id[k]: k for k in user2id}

    item_ids = sorted(df['item_id'].unique())
    n_item = len(item_ids)
    item2id = dict(zip(item_ids, range(start_idx, start_idx + len(item_ids))))
    id2item = {item2id[k]: k for k in item2id}

    vocab['n_user'] = n_user
    vocab['n_item'] = n_item
    vocab['user2id'] = user2id
    vocab['id2user'] = id2user
    vocab['item2id'] = item2id
    vocab['id2item'] = id2item

    return df, vocab


df, vocab = build_vocab(df)
df['user_id'] = df['user_id'].map(vocab['user2id'])
df['item_id'] = df['item_id'].map(vocab['item2id'])
n_user, n_item = vocab['n_user'], vocab['n_item']


def split_data_loo(df, seed=3):
    """分层抽样 / stratified sampling
    Split data into train and test subsets in Leave-One-Out manner.
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#iterating-through-groups
    def samling(df_group, n=1):
        return df_group.sample(n=n)
    df_test = df.groupby('user_id').apply(samling, n=1)
    df_test = df_test.reset_index(drop=True)
    """
    # 使用固定的seed值，保证每次生成的随机划分都相同
    np.random.seed(seed=seed)
    g_train = []
    g_test = []
    grouped = df.groupby('user_id')
    for _, group in grouped:
        # Leave-One-Out
        test = group.sample(n=1)
        # print(test.index)
        train = group[~group.index.isin(test.index)]
        g_train.append(train)
        g_test.append(test)
    df_train = pd.concat(g_train).reset_index(drop=True)
    df_test = pd.concat(g_test).reset_index(drop=True)
    return df_train, df_test


df_train, df_test = split_data_loo(df)

# df.to_hdf(h5_file, key='ratings')
# df = pd.read_hdf(h5_file, key='ratings')
#
# df_train.to_hdf(h5_file, key='train')
# print(df_train.shape, df_train.head())
# n_users = len(df_train['user_id'].unique())
# n_items = len(df_train['item_id'].unique())
# print("train stat: ", n_users, n_items)
#
# df_test.to_hdf(h5_file, key='test')
# print(df_test.shape, df_test.head())
# n_users = len(df_test['user_id'].unique())
# n_items = len(df_test['item_id'].unique())
# print("test stat: ", n_users, n_items)

# csv_file = os.path.join(data_dir, 'ml-1m', 'ratings.csv')
# df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

# csv_file = os.path.join(data_dir, 'ml-1m', 'train.csv')
# df_train.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

# csv_file = os.path.join(data_dir, 'ml-1m', 'test.csv')
# df_test.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

df_neg_sample = sample_negative_df(df)
print(df_neg_sample.shape, df_neg_sample.head())


def load_data_df():
    h5_file = os.path.join(data_dir, 'ml-1m', 'ratings.h5')

    df_train = pd.read_hdf(h5_file, key='train')
    df_test = pd.read_hdf(h5_file, key='test')

    # Testing data with positive items and negative items
    df_neg_sample = pd.read_hdf(h5_file, key='neg_sample')
    df_test = pd.merge(df_test, df_neg_sample, how='left', on='user_id')
    print(df_test.shape, df_test.columns)
    print(df_test.head())

    df_train['rating'] = df_train['rating'].apply(lambda v: 1 if 1 >= 1 else 0)
    df_test['rating'] = df_test['rating'].apply(lambda v: 1 if 1 >= 1 else 0)

    return df_train, df_test


def load_data_csv():
    csv_file = os.path.join(data_dir, 'train.csv')
    df_train = pd.read_csv(csv_file, sep=',', header='infer')

    csv_file = os.path.join(data_dir, 'test.csv')
    df_test = pd.read_csv(csv_file, sep=',', header='infer')

    # Testing data with positive items and negative items
    csv_file = os.path.join(data_dir, 'neg_item.csv')
    df_test_neg = pd.read_csv(csv_file, sep=',', header='infer')

    df_test = pd.merge(df_test, df_test_neg, how='left', on='user_id')
    print(df_test.shape, df_test.columns)
    print(df_test.head())

    df_train['rating'] = df_train['rating'].apply(lambda v: 1 if 1 >= 1 else 0)
    df_test['rating'] = df_test['rating'].apply(lambda v: 1 if 1 >= 1 else 0)

    return df_train, df_test


start = time.time()
mat = df2mat(df, n_user, n_item)
elapsed = time.time() - start
print('finish df2mat, it takes {} seconds.'.format(elapsed))

start = time.time()
train_mat = df2mat(df_train, n_user, n_item)
elapsed = time.time() - start
print('finish df2mat, it takes {} seconds.'.format(elapsed))

start = time.time()
test_mat = df2mat(df_test, n_user, n_item)
elapsed = time.time() - start
print('finish df2mat, it takes {} seconds.'.format(elapsed))

# npz_file = os.path.join(data_dir, 'ml-1m', 'ratings.npz')
# scipy.sparse.save_npz(npz_file, mat)
# # mat = scipy.sparse.load_npz(npz_file)

# npz_file = os.path.join(data_dir, 'ml-1m', 'train.npz')
# scipy.sparse.save_npz(npz_file, mat)
# # mat = scipy.sparse.load_npz(npz_file)

# npz_file = os.path.join(data_dir, 'ml-1m', 'test.npz')
# scipy.sparse.save_npz(npz_file, mat)
# # mat = scipy.sparse.load_npz(npz_file)


def load_data_mat():
    npz_file = os.path.join(data_dir, 'train.npz')
    mat_train = scipy.sparse.load_npz(npz_file)

    npz_file = os.path.join(data_dir, 'test.npz')
    mat_test = scipy.sparse.load_npz(npz_file)

    print(mat_train.shape)
    print(mat_train.shape)

    return mat_train.todok(), mat_test


# users, items = test_mat.nonzero()
negative = sample_negative(mat, test_mat, neg_ratio=99)


def dump_data_ml_1m():
    path = os.path.join(data_dir, 'data.pickle')
    values = [
        n_user, n_item, mat,
        train_mat, test_mat, negative
    ]
    with open(path, 'wb') as f:
        pickle.dump(values, f)
    return values


dump_data_ml_1m()


def load_data_ml_1m():
    path = os.path.join(data_dir, 'data.pickle')
    names = [
        'n_user', 'n_item', 'mat',
        'train_mat', 'test_mat', 'negative'
    ]
    with open(path, 'rb') as f:
        items = pickle.load(f)
    data_dict = dict(zip(names, items))
    return data_dict


data_dict = load_data_ml_1m()

# print(data_dict)
print("# of users: ", data_dict['n_user'])
print("# of items: ", data_dict['n_item'])
print("# of user-item interactions: ", data_dict['mat'].nnz)
