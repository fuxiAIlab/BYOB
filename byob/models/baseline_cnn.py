import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemCNNModel(nn.Module):
    """
    https://github.com/Shawn1993/cnn-text-classification-pytorch
    https://github.com/dennybritz/cnn-text-classification-tf
    https://github.com/facebookresearch/pytext
    """

    def __init__(self, conf):
        super(ItemCNNModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.max_len = conf['max_len']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', 128)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        self.cnn = nn.Conv1d(self.embed_dim, self.hidden_dim, 3, stride=1)
        size = int((self.max_len + 2 * 0 - 1 * (3 - 1) - 1) / 1 + 1)
        # self.pool = nn.MaxPool1d(size)
        self.pool = nn.AvgPool1d(size)
        # nn.init.normal_(self.user_embed.weight, std=0.01)
        # nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)
        self.fc = nn.Linear(self.embed_dim + self.hidden_dim + self.embed_dim, 1)

    def forward(self, inputs):
        u, i, seq = inputs
        # print(u.shape, i.shape)
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, 1, E)
        i = self.item_embed(i.long()).squeeze(dim=1)  # (N, L, E)
        seq = self.item_embed(seq.long())  # (N, L, E)
        seq = seq.permute(0, 2, 1)  # (N, E, L)
        seq = self.cnn(seq)  # (N, H, L)
        h = self.pool(seq).squeeze(-1)  # (N, H)
        # print(seq.shape, h.shape)
        # h = torch.mean(seq, dim=-1)
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

    model = ItemCNNModel(conf).to(device)
    print(model)
    y = model((u, i, seq))
    print(y.size())
