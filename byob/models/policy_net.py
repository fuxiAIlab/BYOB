import logging
import numpy as np
import gym
import time
import math

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class ItemPointer(nn.Module):

    def __init__(self, conf):
        super(ItemPointer, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.bundle_size = conf['bundle_size']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', 128)
        self.encoder = conf.get('encoder', False)
        self.concat = conf.get('concat', False)
        self.num_heads = conf.get('num_heads', 1)
        self.num_layers = conf.get('num_layers', 1)
        self.dropout = conf.get('dropout', 0.1)
        self.clip = conf.get('clip', 10)
        # self.input_dim = conf.get('num_features', 1)
        # self.output_dim = conf.get('num_classes', 1)
        # self.device = conf.get('device', 'cpu')
        self._build()
        if conf.get('embed_path', None):
            self.load_embedding(conf['embed_path'])
            if not conf.get('fine_tune', True):
                self.item_embed.weight.requires_grad = False
            print('-' * 80)
            print(f"load pre-trained embedding: {conf['embed_path']} (fine tune: {conf.get('fine_tune', True)})")
            # print(self.item_embed.weight)
            print('-' * 80)

    def _build(self):

        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        nn.init.xavier_normal_(self.user_embed.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_embed.weight, gain=1.0)

        if self.encoder:
            # encoder_layer = TransformerEncoderLayer(self.embed_dim, self.num_heads, self.hidden_dim, self.dropout)
            # self.seq_encoder = TransformerEncoder(encoder_layer, self.num_layers)
            encoder_layer = TransformerEncoderLayer(self.embed_dim, self.num_heads, self.hidden_dim, self.dropout)
            self.bundle_encoder = TransformerEncoder(encoder_layer, self.num_layers)
            encoder_layer = TransformerEncoderLayer(self.embed_dim, self.num_heads, self.hidden_dim, self.dropout)
            self.pool_encoder = TransformerEncoder(encoder_layer, self.num_layers)
        if self.concat:
            self.fc = nn.Linear(self.bundle_size * self.embed_dim, self.embed_dim)

        self.w1 = nn.Linear(3 * self.embed_dim, self.hidden_dim)
        self.w2 = nn.Conv1d(self.embed_dim, self.hidden_dim, 1, 1)
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_dim), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(self.hidden_dim)), 1. / math.sqrt(self.hidden_dim))

    def forward(self, inputs):
        user, seq, pool, bundle = inputs  # (N,), (N, L), (N, S), (N, K)
        batch_size, pool_size = pool.size(0), pool.size(1)

        user = self.user_embed(user).squeeze(dim=1)  # (N, E)

        seq = self.item_embed(seq).squeeze(dim=1)  # (N, L, E)
        seq = torch.mean(seq, dim=1)  # (N, E)
        # seq = seq.permute(1, 0, 2)  # (L, N, E)
        # seq = self.seq_encoder(seq, mask=None)  # (L, N, E)
        # seq = torch.mean(seq, dim=0)  # (N, E)

        bundle = self.item_embed(bundle).squeeze(dim=1)  # (N, K, E)
        if self.encoder:
            bundle = bundle.permute(1, 0, 2)  # (K, N, E)
            bundle = self.bundle_encoder(bundle, mask=None)  # (K, N, E)
            bundle = bundle.permute(1, 0, 2)  # (N, K, E)
        if self.concat:
            # bundle = F.tanh(self.fc(bundle.view(batch_size, -1)))  # (N, E)
            bundle = F.tanh(self.fc(bundle.reshape(batch_size, -1)))  # (N, E)
        else:
            bundle = torch.mean(bundle, dim=1)  # (N, E)

        pool = self.item_embed(pool).squeeze(dim=1)  # (N, S, E)
        if self.encoder:
            pool = pool.permute(1, 0, 2)  # (S, N, E)
            pool = self.pool_encoder(pool, mask=None)  # (S, N, E)
            pool = pool.permute(1, 0, 2)  # (N, S, E)
        pool = pool.permute(0, 2, 1)  # (N, E, S)
        # pool = torch.mean(pool, dim=1)  # (N, E)

        query = torch.cat([user, seq, bundle], dim=-1)  # (N, 3E)
        features = query

        query = self.w1(query)  # (N, H)
        query = query.unsqueeze(2).repeat(1, 1, pool_size)  # (N, H, S)
        reference = self.w2(pool)  # (N, H, S)
        v = self.v.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # (N, 1, H)
        # print(query.shape, reference.shape, v.shape)
        logits = torch.bmm(v, torch.tanh(query + reference)).squeeze(1)  # (N, S)

        if self.clip:
            logits = self.clip * torch.tanh(logits)

        return logits, features  # (N, S), (N, 3E)

    def predict(self, x):
        pass

    def get_embedding(self):
        return self.item_embed.weight.data.cpu().numpy()

    def set_embedding(self, weight):
        self.item_embed.weight.data.copy_(torch.from_numpy(weight))

    def save_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        a = self.item_embed.weight.data.numpy()
        with open(path, 'wb') as f:
            np.save(f, a)
        return self

    def load_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'rb') as f:
            a = np.load(f)
        self.item_embed.weight.data.copy_(torch.from_numpy(a))
        return self


class PolicyNetwork(TorchModelV2, nn.Module):
    """The policy network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        custom_model_config = model_config.get("custom_model_config", {})
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            pass

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            pass
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation))
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None)
            else:
                self.num_outputs = (
                    [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            pass

        # self._hidden_layers = nn.Sequential(*layers)

        self._hidden_layers = ItemPointer(custom_model_config)
        prev_layer_size = 3 * custom_model_config['embed_dim']  # dimension of state
        # prev_layer_size = 3 * custom_model_config['hidden_dim']  # dimension of state

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            pass

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        # obs = input_dict["obs_flat"].float()
        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        # self._features = self._hidden_layers(self._last_flat_in)
        # logits = self._logits(self._features) if self._logits else self._features

        obs = input_dict["obs_flat"].long()
        user = input_dict["user"].long()
        seq = input_dict["seq"].long()
        pool = input_dict["pool"].long()
        bundle = input_dict["bundle"].long()
        # print(obs.shape, user.shape, seq.shape, pool.shape, bundle.shape)
        # torch.Size([128, 23]) torch.Size([128, 1]) torch.Size([128, 20]) torch.Size([128, 20]) torch.Size([128, 3])
        self._last_flat_in = (user, seq, pool, bundle)
        logits, self._features = self._hidden_layers(self._last_flat_in)
        # print(logits.shape, self._features.shape)

        if self.free_log_std:
            pass
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            pass
        else:
            return self._value_branch(self._features).squeeze(1)


if __name__ == '__main__':

    conf = {
        'n_user': 100,
        'n_item': 3707,
        'seq_len': 20,
        'pool_size': 20,
        'bundle_size': 3,
        'embed_dim': 32,
        'hidden_dim': 128,
        'batch_size': 2,
        'embed_path': None,
        'fine_tune': True,
    }

    user = torch.randint(conf['n_user'], (conf['batch_size'], 1))
    seq = torch.randint(conf['n_item'], (conf['batch_size'], conf['seq_len']))
    pool = torch.randint(conf['n_item'], (conf['batch_size'], conf['pool_size']))
    bundle = torch.randint(conf['n_item'], (conf['batch_size'], conf['bundle_size']))
    print(user.shape, seq.shape, pool.shape, bundle.shape)

    conf['embed_path'] = '/root/reclib/byob/output/models/movielens-SkipGramModel.npy'
    conf['fine_tune'] = False

    model = ItemPointer(conf)
    print(model)

    a = model.get_embedding()
    print(a.shape, a[:3])

    start_time = time.time()
    logits, features = model((user, seq, pool, bundle))
    pointer = torch.softmax(logits, dim=1)
    elapsed = time.time() - start_time
    print(logits.shape, features.shape, elapsed)
    print(logits, pointer.sum(dim=1), sep='\n')
