import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from byob.config import data_dir, model_dir, DATA_CONFIG
from byob.data_utils import setup_dataset_vec
from byob.models.item2vec import SkipGramModel
from byob.utils import read_pickle, as_tensor

conf = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset': 'yoochoose',  # ('movielens', 'yoochoose')
    'context_size': 5,
    'neg_samples': 15,
    'embed_dim': 32,
    'hidden_dim': 64,
    'num_epochs': 10,
    'batch_size': 256,
    'lr': 0.001
}
conf.update(DATA_CONFIG[conf['dataset']])
pkl_file = osp.join(data_dir, conf['dataset'], conf['item_vocab'])
vocab = read_pickle(pkl_file)
conf['vocab_size'] = len(vocab)
assert conf['n_item'] == conf['vocab_size']
print(conf)

# pkl_file = os.path.join(data_dir, conf['dataset'], 'dataset.pkl')
# # write_pickle(pkl_file, dataset)
# dataset = read_pickle(pkl_file)

csv_file = osp.join(data_dir, conf['dataset'], 'seqs.csv')
dataset = setup_dataset_vec(csv_file, vocab, c=conf['context_size'], k=conf['neg_samples'])
loader = DataLoader(dataset, conf['batch_size'], shuffle=True)

model = SkipGramModel(conf).to(conf['device'])
optimizer = torch.optim.SGD(model.parameters(), lr=conf['lr'])

losses = []
for epoch in range(1, conf['num_epochs'] + 1):

    # csv_file = osp.join(data_dir, conf['dataset'], 'seqs.csv')
    # dataset = setup_dataset_vec(csv_file, vocab, c=conf['context_size'], k=conf['neg_samples'])
    # loader = DataLoader(dataset, conf['batch_size'], shuffle=True)

    total_loss = 0.0
    steps_per_epoch = int(len(loader.dataset) / conf['batch_size'])
    for i, batch in enumerate(loader):
        batch = as_tensor(*batch, dtype=torch.long, device=conf['device'])
        # print(batch)
        # model.zero_grad()
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i > 0 and i % 1000 == 0:
            print(f'epoch: {epoch}/{conf["num_epochs"]}, iteration: {i}/{len(loader)}, loss: {loss.item()}')

    file_name = '%s-%s-%d.pt' % (conf['dataset'], type(model).__name__, epoch)
    path = os.path.join(model_dir, file_name)
    model.save_model(path)

    file_name = '%s-%s-%d.npy' % (conf['dataset'], type(model).__name__, epoch)
    path = os.path.join(model_dir, file_name)
    model.save_embedding(path)

    losses.append(total_loss / steps_per_epoch)

print(losses)

file_name = '%s-%s.pt' % (conf['dataset'], type(model).__name__)
path = os.path.join(model_dir, file_name)
model.save_model(path)

file_name = '%s-%s.npy' % (conf['dataset'], type(model).__name__)
path = os.path.join(model_dir, file_name)
model.save_embedding(path)
