data_dir = "/root/reclib/data"
# data_dir = "C:/Users/Desktop/code/reclib/data"
model_dir = "/root/reclib/byob/output/models"
output_dir = "/root/reclib/byob/output"

# path = "/root/reclib"
# # path = r"C:\Users\\Desktop\code\reclib"
# if path not in sys.path:
#     sys.path.append(path)
# print(sys.path)

DATA_CONFIG = {
    'movielens': {
        'n_user': 6040 + 1,
        'n_item': 3706 + 1,
        'start_idx': 1,
        'user_vocab': 'vocab.user.pkl',
        'item_vocab': 'vocab.item.pkl',
        'seq_file': 'seqs.csv',
        'click_file': 'click.csv',
        'buy_file': 'buy.csv',
        'train_item': 'train_item.csv',
        'train_item_rank': 'train_item_rank.csv',
        'train_bundle': 'train_bundle.csv',
        'train_bundle_rank': 'train_bundle_rank.csv',
        'test_bundle': 'test_bundle.csv',
        'max_len': 20,
        'seq_len': 20,
        'min_freq': 5,
        'win_size': 7,
        'top_k': 3
    },
    'yoochoose': {
        'n_user': 60198 + 1,
        'n_item': 33781 + 1,
        'start_idx': 1,
        'user_vocab': 'vocab.user.pkl',
        'item_vocab': 'vocab.item.pkl',
        'seq_file': 'seqs.csv',
        'click_file': 'click_20.csv',
        'buy_file': 'buy_5.csv',
        'train_item': 'train_item.csv',
        'train_item_rank': 'train_item_rank.csv',
        'train_bundle': 'train_bundle.csv',
        'train_bundle_rank': 'train_bundle_rank.csv',
        'test_bundle': 'test_bundle.csv',
        'max_len': 20,
        'seq_len': 20,
        'min_freq': 5,  # 1(38750) / 5 (25844) / 10 (22077)
        'win_size': 7,
        'top_k': 3
    },
    'taobao': {
        'n_user': 6040,
        'n_item': 3706,
        'start_idx': 1,
        'click_file': 'click_20.csv',
        'buy_file': 'buy_3.csv',
        'max_len': 20,
        'seq_len': 20,
        'min_freq': 50,  # 1(3299856) / 20 (512164) / 50 (245108)
        'win_size': 15,
        'top_k': 3
    }
}

MODEL_CONFIG = {
    'BPR': {
        'embed_dim': 32,
    },
    'NCF': {
        'embed_dim': 32,
        'hidden_dim': 128,
        'dropout': 0.0,
    },
    'RNN': {
        'embed_dim': 32,
        'hidden_dim': 128,
        'bidir': True,
        'dropout': 0.0,
    },
    'CNN': {
        'embed_dim': 32,
        'hidden_dim': 128,
        'dropout': 0.0,
    },
    'TRM': {
        'embed_dim': 32,
        'hidden_dim': 128,
        'dropout': 0.0,
    },
    'CBOW': {
        'embed_dim': 32,
        'hidden_dim': 64
    },
    'SG': {
        'embed_dim': 32,
        'hidden_dim': 64
    },
    'BYOB': {
        'embed_dim': 32,
        'hidden_dim': 128,
        'num_heads': 2,
        'num_layers': 1,
        'dropout': 0.1,
        'clip': 10,
    },
}

DEFAULT_CONFIG = {

    # program level
    # ----------------------------------------
    'num_seeds': 1,
    'dataset': 'movielens',
    'model': None,
    'data_dir': '.',
    'output_dir': './output',

    # model specific
    # ----------------------------------------
    'input_dim': 1,  # number of input features
    'output_dim': 1,  # number of output classes
    'embed_dim': 32,  # number of embedding dim
    'hidden_dim': 128,  # number of hidden units
    'd_model': 128,  # dim in Transformer model
    'num_layers': 1,  # hidden layers
    'num_heads': 1,  # attention heads
    'dropout': 0.5,  # dropout probability
    'bidir': False,  # bidirectional in RNN model

    # train options
    # ----------------------------------------
    'num_epochs': 1,
    'batch_size': 128,
    'lr': 1e-3,  # learning rate
    'weight_decay': 1e-5,  # regularization coefficient

    # miscellaneous
    # ----------------------------------------
    'device': 'cpu',  # cpu / cuda
    'verbose': False,
    'ckpt_freq': 1,  # checkpoint frequency (by epoch)
    'log_epochs': 1,
    'log_steps': 100,
    'top_k': 10,
}
