import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemBPRModel(nn.Module):

    def __init__(self, conf):
        super(ItemBPRModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        # nn.init.normal_(self.user_embed.weight, std=0.01)
        # nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

    def _forward(self, inputs):
        u, i, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, E)
        i = self.item_embed(i.long()).squeeze(dim=1)  # (N, E)
        # seq = self.item_embed(seq.long())  # (N, L, E)
        # u = u + torch.mean(seq, dim=1)  # (N, E)
        logits = torch.mul(u, i).sum(dim=-1, keepdim=True)  # (N, 1)
        # logits = (u * i).sum(dim=-1, keepdim=True)
        return torch.sigmoid(logits)

    def forward(self, inputs):
        u, i, j, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, E)
        i = self.item_embed(i.long()).squeeze(dim=1)  # (N, E)
        j = self.item_embed(j.long()).squeeze(dim=1)  # (N, E)
        # seq = self.item_embed(seq.long())  # (N, L, E)
        # u = u + torch.mean(seq, dim=1)  # (N, E)
        pos_logits = torch.mul(u, i).sum(dim=-1, keepdim=True)  # (N, 1)
        neg_logits = torch.mul(u, j).sum(dim=-1, keepdim=True)  # (N, 1)
        # pos_logits = (u * i).sum(dim=-1, keepdim=True)
        # neg_logits = (u * j).sum(dim=-1, keepdim=True)
        return pos_logits, neg_logits

    def predict(self, inputs):
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self._forward(inputs)  # (N, 1)
        pred = pred.detach().cpu().numpy()
        return pred


class BundleBPRModel(nn.Module):

    def __init__(self, conf):
        super(BundleBPRModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        # nn.init.normal_(self.user_embed.weight, std=0.01)
        # nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

    def _forward(self, inputs):
        # [print(v.shape) for v in inputs]
        u, b, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, E)
        b = self.item_embed(b.long()).squeeze(dim=1)  # (N, K, E)
        b = torch.mean(b, dim=1)  # (N, E)
        # seq = self.item_embed(seq.long())  # (N, L, E)
        # u = u + torch.mean(seq, dim=1)  # (N, E)
        logits = torch.mul(u, b).sum(dim=-1, keepdim=True)  # (N, 1)
        # logits = (u * b).sum(dim=-1, keepdim=True)
        return torch.sigmoid(logits)

    def forward(self, inputs):
        u, bi, bj, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, E)
        bi = self.item_embed(bi.long()).squeeze(dim=1)  # (N, K, E)
        bj = self.item_embed(bj.long()).squeeze(dim=1)  # (N, K, E)
        bi = torch.mean(bi, dim=1)  # (N, E)
        bj = torch.mean(bj, dim=1)  # (N, E)
        # seq = self.item_embed(seq.long())  # (N, L, E)
        # u = u + torch.mean(seq, dim=1)  # (N, E)
        pos_logits = torch.mul(u, bi).sum(dim=-1, keepdim=True)  # (N, 1)
        neg_logits = torch.mul(u, bj).sum(dim=-1, keepdim=True)  # (N, 1)
        # pos_logits = (u * bi).sum(dim=-1, keepdim=True)
        # neg_logits = (u * bj).sum(dim=-1, keepdim=True)
        return pos_logits, neg_logits

    def predict(self, inputs):
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self._forward(inputs)  # (N, 1)
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

    model = ItemBPRModel(conf).to(device)
    print(model)
    y_pos, y_neg = model((u, pos, neg, seq))
    print(y_pos.size(), y_neg.size())
    y = model.predict((u, i, seq))
    print(y.shape)
