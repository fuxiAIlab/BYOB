import os
import os.path as osp
import time
import math
import itertools
import random

import numpy as np
import pandas as pd
import torch

import gym
from gym import spaces, logger
from gym.utils import seeding

from byob.config import data_dir, model_dir, DATA_CONFIG, MODEL_CONFIG, DEFAULT_CONFIG
from byob.models.baseline_bpr import ItemBPRModel, BundleBPRModel
from byob.models.baseline_ncf import ItemNCFModel, BundleNCFModel
from byob.models.item2vec import CBOWModel, SkipGramModel
from byob.data_utils import setup_dataset_test
from byob.utils import read_csv, write_csv, read_pickle, write_pickle
from byob.metrics import bundle_metrics


def load_data(conf):

    pkl_file = osp.join(data_dir, conf['dataset'], conf['user_vocab'])
    print("load user vocab:", pkl_file)
    user_vocab = read_pickle(pkl_file)
    pkl_file = osp.join(data_dir, conf['dataset'], conf['item_vocab'])
    print("load item vocab:", pkl_file)
    item_vocab = read_pickle(pkl_file)
    user_set = set([user_vocab[tok] for tok in user_vocab.freqs])
    item_set = set([item_vocab[tok] for tok in item_vocab.freqs])
    # print(len(user_set), min(user_set), max(user_set))
    # print(len(item_set), min(item_set), max(item_set))

    csv_file = osp.join(data_dir, conf['dataset'], conf['seq_file'])
    print("load train data:", csv_file)
    data = read_csv(csv_file)
    # print(len(data), data[0])
    train_ds = dict()
    for user, _, click, buy in data:
        if len(buy) == 0:
            continue
        user = int(user)
        click = list(map(int, click.split('|')))
        buy = list(map(int, buy.split('|')))
        if len(buy) < conf['bundle_size']:
            continue
        train_ds[user] = (click, buy)
    # print(len(train_ds), min(train_ds), max(train_ds))

    csv_file = osp.join(data_dir, conf['dataset'], 'test_bundle_%d_%d.csv' % (conf['pool_size'], conf['bundle_size']))
    print("load test data:", csv_file)
    data = read_csv(csv_file)
    # print(len(data), data[0])
    test_ds = dict()
    for user, pos, pool, seq in data:
        user = int(user)
        pos = list(map(int, pos.split('|')))
        assert len(pos) == conf['bundle_size']
        pool = list(map(int, pool.split('|')))
        seq = list(map(int, seq.split('|')))
        test_ds[user] = (seq, pool, pos)
    # print(len(test_ds), min(test_ds), max(test_ds))

    return train_ds, test_ds, user_set, item_set


def load_model(conf):

    conf.update(MODEL_CONFIG[conf['item_model']])
    if conf['item_model'] == 'BPR':
        item_model = ItemBPRModel(conf).to(conf['device'])
    else:
        item_model = ItemNCFModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(item_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load item model:", path)
    item_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    item_model.eval()

    conf.update(MODEL_CONFIG[conf['bundle_model']])
    if conf['bundle_model'] == 'BPR':
        bundle_model = BundleBPRModel(conf).to(conf['device'])
    else:
        bundle_model = BundleNCFModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(bundle_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load bundle model:", path)
    bundle_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    bundle_model.eval()

    conf.update(MODEL_CONFIG[conf['compat_model']])
    compat_model = SkipGramModel(conf).to(conf['device'])
    file_name = '%s-%s-10.pt' % (conf['dataset'], type(compat_model).__name__)
    path = osp.join(model_dir, file_name)
    print("load compat model:", path)
    compat_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    compat_model.eval()

    return item_model, compat_model, bundle_model


class BundleEnv(gym.Env):
    """
    Description:
        Bundle Composition Environment.

    Observation:
        Type: Box()
        Num	Observation    Min  Max
        u	User Features  -Inf  Inf
        G	Candidate Items  -Inf  Inf
        ...

    Actions:
        Type: Discrete(N)
        Num	Action
        0	Choose Item 0
        1	Choose Item 1
        ...

    Reward:
        Reward is evaluate at each step.

    Starting State:
        The bundle is empty.

    Episode Termination:
        The whole bundle are formed.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
    }

    pool_size = 20  # number of candidate items
    bundle_size = 3  # number of items per bundle
    num_features = 1  # number of user/item features

    def __init__(self, conf):

        self.num_users = conf['n_user']  # number of all users
        self.num_items = conf['n_item']  # number of all items
        self.start_idx = conf['start_idx']  # start index of users/items
        self.seq_len = conf['seq_len']  # length of behaviour sequence
        self.pool_size = conf.get('pool_size', self.pool_size)
        self.bundle_size = conf.get('bundle_size', self.bundle_size)
        self.num_features = conf.get('num_features', self.num_features)
        self.env_mode = conf.get('env_mode', 'train')  # train/test
        self.rew_mode = conf.get('rew_mode', 'item')  # item/compat/bundle/metric
        self.metric = conf.get('metric', 'recall')  # precision/precision_plus/recall

        self.train_ds, self.test_ds, self.user_set, self.item_set = load_data(conf)
        # assert self.num_users == self.start_idx + len(self.train_ds)
        self.item_model, self.compat_model, self.bundle_model = load_model(conf)

        # # feature array for item pool (pool_size, num_features)
        # self.x = None
        # # whether sort items by feature or not to ensure consistent input order
        # self.sort = True

        high = 999999
        self.action_space = spaces.Discrete(self.pool_size)
        # self.observation_space = spaces.Box(-1.0, 1.0, shape=(2 * self.pool_size * self.num_features,))
        self.observation_space = spaces.Dict({
            "user": spaces.Box(-high, high, shape=(1,), dtype=np.int32),
            "seq": spaces.Box(-high, high, shape=(self.seq_len,), dtype=np.int32),
            "pool": spaces.Box(-high, high, shape=(self.pool_size,), dtype=np.int32),
            "pos": spaces.Box(-high, high, shape=(self.bundle_size,), dtype=np.int32),
            "mask": spaces.Box(0, 1, shape=(self.pool_size,), dtype=np.int32),
            "bundle": spaces.Box(-high, high, shape=(self.bundle_size,), dtype=np.int32),
            "state": spaces.Box(-high, high, shape=(self.pool_size + self.bundle_size,), dtype=np.int32),
        })
        self.reward_range = (-float('inf'), float('inf'))

        self.state = None
        self.seed()

        self.max_episode_steps = self.pool_size
        self.elapsed_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reward(self, state, action):
        assert self.elapsed_steps <= self.bundle_size
        user, seq, pool, pos, mask, bundle = state
        # print(state)
        # seq = seq[np.newaxis, :]  # (1, L)
        num_selected = np.count_nonzero(bundle)
        # item preference reward
        # ------------------------------------------------------------------------
        r1 = 0.0
        if 'item' in self.rew_mode:
            item = np.array([pool[action]])  # (1,)
            r1 = self.item_model.predict((user[np.newaxis], item[np.newaxis], seq[np.newaxis]))
            # print(r1.shape, r1, np.ravel(r1))
            r1 = np.ravel(r1)[0]
        # item compatibility reward
        # ------------------------------------------------------------------------
        r2 = 0.0
        if 'compat' in self.rew_mode:
            if num_selected > 0:
                inp = np.array([pool[action]] * num_selected).reshape(-1, 1)
                out = bundle[bundle > 0].reshape(-1, 1)
                assert inp.shape == out.shape
                r2 = self.compat_model.predict((inp, out))
                # print(r2.shape, r2, np.ravel(r2))
                r2 = np.ravel(r2.sum())[0]
        # bundle preference reward
        # ------------------------------------------------------------------------
        r3 = 0.0
        if 'bundle' in self.rew_mode:
            if num_selected >= self.bundle_size - 1:
                pred = np.array(bundle)  # (K,)
                pred[-1] = pool[action]
                r3 = self.bundle_model.predict((user[np.newaxis], pred[np.newaxis], seq[np.newaxis]))
                # print(r3.shape, r3, np.ravel(r3))
                r3 = np.ravel(r3)[0]
        # ------------------------------------------------------------------------
        # metric reward
        # ------------------------------------------------------------------------
        r4 = 0.0
        if 'metric' in self.rew_mode:
            if num_selected >= self.bundle_size - 1:
                pred = np.array(bundle)  # (K,)
                pred[-1] = pool[action]  # (K,)
                r4 = bundle_metrics(pos[np.newaxis], pred[np.newaxis])
                # print(pos, pred, r4)
                # r4 = r4['precision'] + r4['precision_plus'] + r4['recall']
                r4 = r4[self.metric]
        # ------------------------------------------------------------------------
        # print(r1, r2, r3, r4, num_selected, pred, pool[action])
        return r1 + r2 + r3 + r4

    def step(self, action):
        assert self.elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
        self.elapsed_steps += 1
        # reward function
        # ------------------------------------------------------------------------
        start_time = time.time()
        reward = self._reward(self.state, action)
        elapsed = time.time() - start_time
        # transition dynamics
        # ------------------------------------------------------------------------
        user, seq, pool, pos, mask, bundle = self.state
        # 1 <= elapsed_steps <= bundle_size
        bundle[self.elapsed_steps - 1] = pool[action]
        # 0 <= n_chosed <= bundle_size - 1
        # n_chosed = self.elapsed_steps % self.bundle_size
        # bundle[n_chosed - 1] = pool[action]
        pool[action] = 0
        mask[action] = 0
        self.state = (user, seq, pool, pos, mask, bundle)
        # terminal state
        # ------------------------------------------------------------------------
        # done = True if self.elapsed_steps >= self.max_episode_steps else False
        done = True if self.elapsed_steps >= self.bundle_size else False
        if done:
            # print("bundle composition: ", self.elapsed_steps, bundle, reward)
            pass
        # return np.array(self.state), reward, done, {}
        # return np.array(np.concatenate(self.state)), reward, done, {}
        # return list(map(np.array, self.state)), reward, done, {}
        return {
            "user": np.array(user),
            "seq": np.array(seq),
            "pool": np.array(pool),
            "pos": np.array(pos),
            "mask": np.array(mask),
            "bundle": np.array(bundle),
            "state": np.concatenate((pool, bundle))
        }, reward, done, {"elapsed": elapsed}

    def _train_state(self):
        # user identity
        # -----------------------------------------------------------------
        # user = np.random.choice(list(self.user_set), 1)
        user = np.random.choice(list(self.train_ds.keys()), 1)
        click, buy = self.train_ds[user.item()]
        # historical behaviors
        # -----------------------------------------------------------------
        # seq = random.sample(click, k=seq_len)
        idx = np.random.randint(0, len(click) - self.seq_len + 1)
        seq = click[idx:idx + self.seq_len]
        # positive bundle
        # -----------------------------------------------------------------
        # pos = random.sample(buy, k=K)
        # pos = np.random.choice(buy, size=K, replace=False)
        idx = np.random.randint(0, len(buy) - self.bundle_size + 1)
        pos = buy[idx:idx + self.bundle_size]
        random.shuffle(pos)
        # candidate items
        # -----------------------------------------------------------------
        # pool = np.random.choice(list(self.item_set), self.pool_size, replace=False)
        neg_set = self.item_set - set(buy)
        neg = random.sample(neg_set, k=self.pool_size - self.bundle_size)
        pool = pos + neg  # candidate items
        random.shuffle(pool)
        return map(np.array, (user, seq, pool, pos))

    def _test_state(self):
        user = np.random.choice(list(self.test_ds.keys()), 1)
        seq, pool, pos = self.test_ds[user.item()]
        return map(np.array, (user, seq, pool, pos))

    def reset(self):
        if self.env_mode == 'train':
            user, seq, pool, pos = self._train_state()
            # user, seq, pool, pos = self._test_state()  # over-fitting
        else:
            user, seq, pool, pos = self._test_state()
        mask = np.array([1] * self.pool_size, dtype=np.int32)
        bundle = np.zeros(shape=(self.bundle_size,), dtype=np.int32)
        # (1,) (20,) (20,) (3,) (20,) (3,) all the data type is  <class 'numpy.ndarray'>
        # print(type(user), type(seq), type(pool), type(pos), type(mask), type(bundle))
        # print(user.shape, seq.shape, pool.shape, pos.shape, mask.shape, bundle.shape)
        self.state = (user, seq, pool, pos, mask, bundle)
        self.elapsed_steps = 0
        # return np.array(self.state)
        # return np.array(np.concatenate(self.state))
        # return list(map(np.array, self.state))
        return {
            "user": np.array(user),
            "seq": np.array(seq),
            "pool": np.array(pool),
            "pos": np.array(pos),
            "mask": np.array(mask),
            "bundle": np.array(bundle),
            "state": np.concatenate((pool, bundle))
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':

    conf = {
        'dataset': 'movielens',
        'pool_size': 20,
        'bundle_size': 3,
        'item_model': 'NCF',
        'bundle_model': 'NCF',
        'compat_model': 'SG',
        'num_features': 1,
        'device': 'cpu'
    }
    conf.update(DATA_CONFIG[conf['dataset']])
    conf['vocab_size'] = conf['n_item']
    conf['env_mode'] = 'train'  # train/test
    conf['rew_mode'] = 'item'  # item/compat/bundle/metric
    conf['metric'] = 'recall'  # precision/precision_plus/recall
    print(conf)

    env = BundleEnv(conf)
    state = env.reset()
    print("Initial State: \n", state)

    pool_size = env.pool_size
    actions = np.random.permutation(range(0, pool_size))
    # actions = list(itertools.permutations(range(1, pool_size + 1), 1))
    print("Test actions: \n", actions)
    print('-' * 80)

    for i in range(pool_size):
        env.render()
        # action = np.random.randint(low=1, high=8 + 1)  # this takes random actions
        action = actions[i]
        next_state, reward, done, info = env.step(action)
        print("Env step %d:" % (i + 1), action, reward, done, info)
        # print("Env step %d:" % (i + 1), state, action, next_state, reward, done)
        # print("Env step %d:" % (i + 1), state, action, next_state, reward, done, sep='\n', end='\n')
        print('-' * 80)
        state = next_state
        if done:
            state = env.reset()
            break
    env.close()
