import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation, output_activation=nn.Identity):
    assert len(sizes) >= 2
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class ItemNCFModel(nn.Module):
    """
    https://github.com/hexiangnan/neural_collaborative_filtering
    https://github.com/microsoft/recommenders
    """

    def __init__(self, conf):
        super(ItemNCFModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', 64)
        self.num_layers = conf.get('num_layers', 1)
        self._build()
        if conf.get('embed_path', None):
            self.load_embedding(conf['embed_path'])

    def _build(self):
        self.user_embed_gmf = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed_gmf = nn.Embedding(self.n_item, self.embed_dim)
        self.user_embed_mlp = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed_mlp = nn.Embedding(self.n_item, self.embed_dim)
        nn.init.xavier_normal_(self.user_embed_gmf.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_embed_gmf.weight, gain=1.0)
        nn.init.xavier_normal_(self.user_embed_mlp.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_embed_mlp.weight, gain=1.0)
        sizes = [2 * self.embed_dim] + [self.hidden_dim, self.embed_dim]
        # sizes = [2 * self.embed_dim] + [self.hidden_dim, self.embed_dim, 1]
        self.mlp = mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity)
        self.fc = nn.Linear(2 * self.embed_dim, 1)

    def forward(self, inputs):
        u, i, _ = inputs
        # print(u, i, u.max(), u.min(), i.max(), i.min())
        u_gmf = self.user_embed_gmf(u).squeeze(dim=1)  # (N, E)
        i_gmf = self.item_embed_gmf(i).squeeze(dim=1)  # (N, E)
        h_gmf = torch.mul(u_gmf, i_gmf)  # (N, E)
        u_mlp = self.user_embed_mlp(u).squeeze(dim=1)  # (N, E)
        i_mlp = self.item_embed_mlp(i).squeeze(dim=1)  # (N, E)
        h_mlp = torch.cat([u_mlp, i_mlp], dim=-1)
        h_mlp = self.mlp(h_mlp)  # (N, E)
        h = torch.cat([h_gmf, h_mlp], dim=-1)  # (N, 2H)
        # probas = torch.sigmoid(self.fc(h))
        logits = self.fc(h)  # (N, 1)
        return torch.sigmoid(logits)

    def predict(self, inputs):
        # [print(v.shape) for v in inputs]
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self.forward(inputs)  # (N, 1)
        pred = pred.detach().cpu().numpy()
        return pred

    def get_embedding(self):
        return self.item_embed_gmf.weight.data.cpu().numpy()

    def set_embedding(self, weight):
        weight = torch.from_numpy(weight)
        self.item_embed_gmf.weight.data.copy_(weight)

    def save_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        a = self.item_embed_gmf.weight.data.numpy()
        a = self.item_embed_mlp.weight.data.numpy()
        with open(path, 'wb') as f:
            np.save(f, a)
        return self

    def load_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'rb') as f:
            a = np.load(f)
        weight = torch.from_numpy(a)
        self.item_embed_gmf.weight.data.copy_(weight)
        self.item_embed_mlp.weight.data.copy_(weight)
        return self


class BundleNCFModel(nn.Module):
    """
    https://github.com/hexiangnan/neural_collaborative_filtering
    https://github.com/microsoft/recommenders
    """

    def __init__(self, conf):
        super(BundleNCFModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', 128)
        self.num_layers = conf.get('num_layers', 1)
        self._build()
        if conf.get('embed_path', None):
            self.load_embedding(conf['embed_path'])

    def _build(self):
        self.user_embed_gmf = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed_gmf = nn.Embedding(self.n_item, self.embed_dim)
        self.user_embed_mlp = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed_mlp = nn.Embedding(self.n_item, self.embed_dim)
        nn.init.xavier_normal_(self.user_embed_gmf.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_embed_gmf.weight, gain=1.0)
        nn.init.xavier_normal_(self.user_embed_mlp.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_embed_mlp.weight, gain=1.0)
        sizes = [2 * self.embed_dim] + [self.hidden_dim, self.embed_dim]
        # sizes = [2 * self.embed_dim] + [self.hidden_dim, self.embed_dim, 1]
        self.mlp = mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity)
        self.fc = nn.Linear(2 * self.embed_dim, 1)

    def forward(self, inputs):
        u, b, _ = inputs
        # [print(v.shape) for v in inputs]
        u_gmf = self.user_embed_gmf(u).squeeze(dim=1)  # (N, E)
        b_gmf = self.item_embed_gmf(b).squeeze(dim=1)  # (N, K, E)
        b_gmf = torch.mean(b_gmf, dim=1)  # (N, E)
        h_gmf = torch.mul(u_gmf, b_gmf)  # (N, E)
        u_mlp = self.user_embed_mlp(u).squeeze(dim=1)  # (N, E)
        b_mlp = self.item_embed_mlp(b).squeeze(dim=1)  # (N, K, E)
        b_mlp = torch.mean(b_mlp, dim=1)  # (N, E)
        h_mlp = torch.cat([u_mlp, b_mlp], dim=-1)
        h_mlp = self.mlp(h_mlp)
        h = torch.cat([h_gmf, h_mlp], dim=-1)  # (N, 2H)
        # probas = torch.sigmoid(self.fc(h))
        logits = self.fc(h)  # (N, 1)
        return torch.sigmoid(logits)

    def predict(self, inputs):
        # [print(v.shape) for v in inputs]
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self.forward(inputs)  # (N, 1)
        pred = pred.detach().cpu().numpy()
        return pred

    def get_embedding(self):
        return self.item_embed_gmf.weight.data.cpu().numpy()

    def set_embedding(self, weight):
        weight = torch.from_numpy(weight)
        self.item_embed_gmf.weight.data.copy_(weight)

    def save_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        a = self.item_embed_gmf.weight.data.numpy()
        a = self.item_embed_mlp.weight.data.numpy()
        with open(path, 'wb') as f:
            np.save(f, a)
        return self

    def load_embedding(self, path):
        path = path + '.npy' if path[-4:] != '.npy' else path
        with open(path, 'rb') as f:
            a = np.load(f)
        weight = torch.from_numpy(a)
        self.item_embed_gmf.weight.data.copy_(weight)
        self.item_embed_mlp.weight.data.copy_(weight)
        return self


if __name__ == '__main__':

    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conf = dict()
    conf['num_features'] = 1
    conf['num_classes'] = 2
    conf['n_user'] = 100
    conf['n_item'] = 10
    conf['max_len'] = 20
    conf['embed_dim'] = 32
    conf['hidden_dim'] = 32
    conf['device'] = 'cpu'
    conf['batch_size'] = 16
    print(conf)

    # x = torch.randn(conf['batch_size'], conf['num_features']).to(device)
    # y = torch.randint(conf['num_classes'], (conf['batch_size'],)).to(device)
    u = torch.randint(conf['n_user'], (conf['batch_size'], 1)).to(device)
    # i = torch.randint(conf['n_item'], (conf['batch_size'], 1)).to(device)
    b = torch.randint(conf['n_item'], (conf['batch_size'], 3)).to(device)
    pos = torch.randint(conf['n_item'], (conf['batch_size'], 1)).to(device)
    neg = torch.randint(conf['n_item'], (conf['batch_size'], 1)).to(device)
    seq = torch.randint(conf['n_item'], (conf['batch_size'], conf['max_len'])).to(device)
    # y = torch.randint(2, (conf['batch_size'],))

    # model = ItemNCFModel(conf).to(device)
    # y = model((u, i, seq))
    model = BundleNCFModel(conf).to(device)
    y = model((u, b, seq))
    print(model)
    print(y.size())
