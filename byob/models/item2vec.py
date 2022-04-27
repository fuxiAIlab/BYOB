import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWModel(nn.Module):
    """
    https://samaelchen.github.io/word2vec_pytorch/
    https://github.com/jojonki/word2vec-pytorch
    """

    def __init__(self, conf):
        super(CBOWModel, self).__init__()
        self.vocab_size = conf['vocab_size']
        self.context_size = conf['context_size']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', None)
        self._build()

    def _build(self):
        self.inp_embed = nn.Embedding(self.vocab_size, self.embed_dim, sparse=True)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_dim, sparse=True)
        if self.hidden_dim:
            self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.inp_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed.weight.data.uniform_(-init_range, init_range)
        if self.hidden_dim:
            self.fc1.weight.data.normal_(-init_range, init_range)
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.normal_(-init_range, init_range)
            self.fc2.bias.data.zero_()
        # nn.init.xavier_normal_(self.inp_embed.weight, gain=1.0)
        # nn.init.xavier_normal_(self.out_embed.weight, gain=1.0)

    def forward(self, inputs):
        inp, out, neg = inputs
        assert inp.size(1) == 2 * self.context_size
        inp_emb = self.inp_embed(inp).squeeze()  # (N, 2C, E)
        # inp_emb = torch.sum(inp_emb, dim=1)  # (N, E)
        inp_emb = torch.mean(inp_emb, dim=1)  # (N, E)
        out_emb = self.out_embed(out).squeeze()  # (N, E)
        if self.hidden_dim:
            inp_emb = F.relu(self.fc1(inp_emb))
            inp_emb = F.relu(self.fc2(inp_emb))
        score = torch.mul(inp_emb, out_emb)  # (N, E)
        score = torch.sum(score, dim=1)  # (N,)
        score = F.logsigmoid(score)
        neg_emb = self.out_embed(neg)  # (N, K, E)
        neg_score = torch.bmm(neg_emb, inp_emb.unsqueeze(2)).squeeze()  # (N, K)
        neg_score = F.logsigmoid(-1 * neg_score)
        neg_score = torch.sum(neg_score, dim=1)  # (N,)
        # return -1 * (torch.sum(score) + torch.sum(neg_score))
        return -1 * (torch.mean(score) + torch.mean(neg_score))

    def predict(self, inputs):
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        # pred = self.forward(inputs)
        inp, out = inputs
        inp_emb = self.inp_embed(inp).squeeze()  # (N, C, E)
        # inp_emb = torch.sum(inp_emb, dim=1)  # (N, E)
        inp_emb = torch.mean(inp_emb, dim=1)  # (N, E)
        out_emb = self.out_embed(out).squeeze()  # (N, E)
        if self.hidden_dim:
            inp_emb = F.relu(self.fc1(inp_emb))
            inp_emb = F.relu(self.fc2(inp_emb))
        score = torch.mul(inp_emb, out_emb)  # (N, E)
        score = torch.sum(score, dim=1)  # (N,)
        pred = torch.sigmoid(score)
        pred = pred.detach().cpu().numpy()
        return pred

    def get_embedding(self):
        return self.inp_embed.weight.data.numpy()

    def set_embedding(self, weight):
        weight = torch.from_numpy(weight)
        self.inp_embed.weight.data.copy_(weight)

    def save_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'wb') as f:
            np.save(f, self.get_embedding())
        return self

    def load_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'rb') as f:
            self.set_embedding(np.load(f))
        return self

    def save_model(self, path):
        path = path + '.pt' if path[-3:] != '.pt' and path[-4:] != '.pth' else path
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        path = path + '.pt' if path[-3:] != '.pt' and path[-4:] != '.pth' else path
        self.load_state_dict(torch.load(path))
        self.eval()
        return self


class SkipGramModel(nn.Module):
    """
    https://adoni.github.io/2017/11/08/word2vec-pytorch/
    https://cloud.tencent.com/developer/article/1613950
    https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    https://github.com/facebookresearch/fastText
    """

    def __init__(self, conf):
        super(SkipGramModel, self).__init__()
        self.vocab_size = conf['vocab_size']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', None)
        self._build()

    def _build(self):
        self.inp_embed = nn.Embedding(self.vocab_size, self.embed_dim, sparse=True)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_dim, sparse=True)
        if self.hidden_dim:
            self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.inp_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed.weight.data.uniform_(-init_range, init_range)
        self.inp_embed.weight.data[0].zero_()
        self.out_embed.weight.data[0].zero_()
        if self.hidden_dim:
            self.fc1.weight.data.normal_(-init_range, init_range)
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.normal_(-init_range, init_range)
            self.fc2.bias.data.zero_()
        # nn.init.xavier_normal_(self.inp_embed.weight, gain=1.0)
        # nn.init.xavier_normal_(self.out_embed.weight, gain=1.0)

    def forward(self, inputs):
        inp, out, neg = inputs
        inp_emb = self.inp_embed(inp).squeeze()  # (N, E)
        out_emb = self.out_embed(out).squeeze()  # (N, E)
        if self.hidden_dim:
            inp_emb = F.relu(self.fc1(inp_emb))
            inp_emb = F.relu(self.fc2(inp_emb))
        score = torch.mul(inp_emb, out_emb)  # (N, E)
        score = torch.sum(score, dim=1)  # (N,)
        score = F.logsigmoid(score)
        neg_emb = self.out_embed(neg)  # (N, K, E)
        neg_score = torch.bmm(neg_emb, inp_emb.unsqueeze(2)).squeeze()  # (N, K)
        neg_score = F.logsigmoid(-1 * neg_score)
        neg_score = torch.sum(neg_score, dim=1)  # (N,)
        # return -1 * (torch.sum(score) + torch.sum(neg_score))
        return -1 * (torch.mean(score) + torch.mean(neg_score))

    def predict(self, inputs):
        # [print(v.shape) for v in inputs]
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        # pred = self.forward(inputs)
        inp, out = inputs
        inp_emb = self.inp_embed(inp).squeeze(dim=1)  # (N, E)
        out_emb = self.out_embed(out).squeeze(dim=1)  # (N, E)
        if self.hidden_dim:
            inp_emb = F.relu(self.fc1(inp_emb))
            inp_emb = F.relu(self.fc2(inp_emb))
        score = torch.mul(inp_emb, out_emb)  # (N, E)
        score = torch.sum(score, dim=1)  # (N,)
        pred = torch.sigmoid(score)
        pred = pred.detach().cpu().numpy()
        return pred

    def get_embedding(self):
        return self.inp_embed.weight.data.cpu().numpy()

    def set_embedding(self, weight):
        self.inp_embed.weight.data.copy_(torch.from_numpy(weight))

    def save_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'wb') as f:
            np.save(f, self.get_embedding())
        return self

    def load_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'rb') as f:
            self.set_embedding(np.load(f))
        return self

    def save_model(self, path):
        path = path + '.pt' if path[-3:] != '.pt' and path[-4:] != '.pth' else path
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        path = path + '.pt' if path[-3:] != '.pt' and path[-4:] != '.pth' else path
        self.load_state_dict(torch.load(path))
        self.eval()
        return self


if __name__ == '__main__':

    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conf = dict()
    conf['vocab_size'] = 100
    conf['context_size'] = 5
    conf['embed_dim'] = 32
    # conf['hidden_dim'] = 64
    conf['hidden_dim'] = None
    conf['device'] = 'cpu'
    conf['batch_size'] = 16
    print(conf)

    # inp = torch.randint(conf['vocab_size'], (conf['batch_size'], 2 * conf['context_size'])).to(device)
    inp = torch.randint(conf['vocab_size'], (conf['batch_size'], 1)).to(device)
    pos = torch.randint(conf['vocab_size'], (conf['batch_size'], 1)).to(device)
    neg = torch.randint(conf['vocab_size'], (conf['batch_size'], 10)).to(device)

    # model = CBOWModel(conf).to(device)
    model = SkipGramModel(conf).to(device)
    print(model)
    y = model((inp, pos, neg))
    print(y.shape)
    y = model.predict((inp, pos))
    print(y.shape)

    a = model.get_embedding()
    print(a.shape, a[:3])

    from byob.config import model_dir
    file_name = '%s-%s.pt' % ('movielens', type(model).__name__)
    path = os.path.join(model_dir, file_name)
    model.save_model(path)
    # model = model.load_model(path)
