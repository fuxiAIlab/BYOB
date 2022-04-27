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


class ItemRNNModel(nn.Module):

    def __init__(self, conf):
        super(ItemRNNModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', 128)
        self.num_layers = conf.get('num_layers', 1)
        self.dropout = conf.get('dropout', 0.0)
        self.bidir = conf.get('bidir', False)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers,
                            batch_first=False, dropout=self.dropout, bidirectional=self.bidir)
        # nn.init.normal_(self.user_embed.weight, std=0.01)
        # nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)
        self.h0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        # sizes = [2 * self.embed_dim] + [self.hidden_dim] * 2
        # # sizes = [2 * self.embed_dim] + [self.hidden_dim] * 2 + [1]
        # self.mlp = mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity)
        self.fc = nn.Linear(self.embed_dim + self.hidden_dim + self.embed_dim, 1)

    def _init_state(self, batch_size):
        r"""Initialize hidden state and cell state.
        - inputs: batch_size ()
        - output: (h0, c0) (num_layers * num_dirs, N, H), initial state
        """
        num_dirs = 2 if self.bidir else 1
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(
            self.num_layers * num_dirs, batch_size, self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(
            self.num_layers * num_dirs, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, inputs):
        u, i, seq = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, 1, E)
        i = self.item_embed(i.long()).squeeze(dim=1)  # (N, L, E)
        seq = self.item_embed(seq.long())  # (N, L, E)
        seq = seq.permute(1, 0, 2)  # (L, N, E)
        h0_c0 = self._init_state(seq.size(1))
        output, hn_cn = self.lstm(seq, h0_c0)  # (L, N, n_directions * H), (num_layers * num_dirs, N, H)
        # output = output.permute(1, 0, 2)  # (N, L, num_dirs * H)
        h = output[-1, :, :]  # (N, num_dirs * H)
        if self.bidir:
            h = h.view(-1, 2, self.hidden_dim)  # (N, 2, H)
            h = torch.mean(h, dim=1)  # (N, H)
        # u = u + h  # (N, E)
        # logits = torch.mul(u, i).sum(dim=-1, keepdim=True)  # (N, 1)
        # logits = (u * i).sum(dim=-1, keepdim=True)
        h = torch.cat([u, h, i], dim=-1)
        logits = self.fc(h)  # (N, 1)
        return torch.sigmoid(logits)

    def predict(self, inputs):
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self.forward(inputs)
        pred = pred.detach().cpu().numpy()
        return pred


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conf = dict()
    conf['num_features'] = 1
    conf['num_classes'] = 2
    conf['n_user'] = 100
    conf['n_item'] = 10
    conf['max_len'] = 20
    conf['embed_dim'] = 32
    conf['hidden_dim'] = 32
    conf['bidir'] = True
    conf['device'] = device
    conf['batch_size'] = 16
    print(conf)

    # x = torch.randn(conf['batch_size'], conf['num_features']).to(device)
    # y = torch.randint(conf['num_classes'], (conf['batch_size'],)).to(device)
    u = torch.randint(conf['n_user'], (conf['batch_size'], 1)).to(device)
    i = torch.randint(conf['n_item'], (conf['batch_size'], 1)).to(device)
    pos = torch.randint(conf['n_item'], (conf['batch_size'], 1)).to(device)
    neg = torch.randint(conf['n_item'], (conf['batch_size'], 1)).to(device)
    seq = torch.randint(conf['n_item'], (conf['batch_size'], conf['max_len'])).to(device)
    # y = torch.randint(2, (conf['batch_size'],))

    model = ItemRNNModel(conf).to(device)
    print(model)
    y = model((u, i, seq))
    print(y.size())
