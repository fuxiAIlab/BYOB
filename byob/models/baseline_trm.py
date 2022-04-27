import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTRMModel(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/Kyubyong/transformer
    https://github.com/huggingface/transformers
    """

    def __init__(self, conf):
        super(ItemTRMModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.max_len = conf['max_len']
        self.embed_dim = conf.get('embed_dim', 32)
        self.hidden_dim = conf.get('hidden_dim', 128)
        self.num_layers = conf.get('num_layers', 1)
        self.num_heads = conf.get('num_heads', 2)
        self.dropout = conf.get('dropout', 0.5)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        # nn.init.normal_(self.user_embed.weight, std=0.01)
        # nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, self.hidden_dim, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        self.fc = nn.Linear(self.embed_dim + self.embed_dim + self.embed_dim, 1)

    def forward(self, inputs):
        u, i, seq = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)  # (N, 1, E)
        i = self.item_embed(i.long()).squeeze(dim=1)  # (N, L, E)
        seq = self.item_embed(seq.long())  # (N, L, E)
        seq = seq.permute(1, 0, 2)  # (L, N, E)
        seq = self.transformer_encoder(seq, mask=None)  # (L, N, E)
        seq = seq.permute(1, 0, 2)  # (N, L, E)
        h = torch.mean(seq, dim=1)  # (N, E)
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

    model = ItemTRMModel(conf).to(device)
    print(model)
    y = model((u, i, seq))
    print(y.size())
