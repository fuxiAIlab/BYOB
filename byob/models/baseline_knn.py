import os

import numpy as np
import scipy.sparse as sp
import pandas as pd


class KNNModel(object):
    """
    KNN Model for Sequential Recommendation
    """

    def __init__(self, config):
        super(KNNModel, self).__init__()

        self.model_type = 'KNN'

        self.k = config['rank_k']
        self.shuffle = config['shuffle']
        self.mask = config['mask']

        self.mat = None
        self.knn = None

    def fit(self, mat, k=5):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
        """
        if k is None:
            k = self.k

        self.mat = mat

        rowsum = np.array(mat.sum(axis=1), dtype=np.float32)
        colsum = np.array(mat.sum(axis=0), dtype=np.float32)
        rowsum[rowsum == 0] = np.inf
        colsum[colsum == 0] = np.inf
        d_inv_row = np.power(rowsum, -0.5).flatten()
        d_inv_col = np.power(colsum, -0.5).flatten()
        d_mat_inv_row = sp.diags(d_inv_row)
        d_mat_inv_col = sp.diags(d_inv_col)

        # sim_mat = mat.T.dot(mat)
        normalized_mat = d_mat_inv_row.dot(mat).dot(d_mat_inv_col)
        sim_mat = normalized_mat.T.dot(normalized_mat)

        sim_mat = sim_mat.toarray()
        for i in range(sim_mat.shape[0]):
            sim_mat[i, i] = -np.inf
        rank_mat = np.argsort(-sim_mat, axis=1)
        self.knn = rank_mat[:, :k]

        return rank_mat

    def predict(self, u, i):
        items = self.mat[u].nonzero()[1]
        for i in items:
            if i in self.knn[i]:
                return True
        return False

    def recommend(self, u, i, k=5):
        assert k <= len(self.knn[0])
        knn = np.random.permutation(self.knn[i])  # random shuffle
        top_k = knn[:k]
        if not self.shuffle:
            return np.sort(top_k)
        return np.array(top_k)


if __name__ == "__main__":

    # data_dir = "/root/reclib/data"
    data_dir = "/data"

    csv_file = os.path.join(data_dir, 'ml-1m', 'ratings.dat')
    h5_file = os.path.join(data_dir, 'ml-1m', 'ratings.h5')

    columns = ['user_id', 'item_id', 'rating', 'ts']
    df = pd.read_csv(csv_file, sep='::', header=None, names=columns)
    # df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')
    # df = pd.read_csv(csv_file, sep=',', header='infer')

    print(df.shape, df.columns)
    print(df.head())

    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

    print("# of ratings: %d" % len(df))
    print("# of users: %d" % len(df['user_id'].unique()))
    print("# of movies: %d" % len(df['item_id'].unique()))

    from byob.utils import build_vocab_df, df2mat

    df, vocab = build_vocab_df(df, min_freq=1)

    df['user_id'] = df['user_id'].map(vocab['user2id'])
    df['item_id'] = df['item_id'].map(vocab['item2id'])

    mat = df2mat(df, vocab)

    model = KNNModel()

    model.fit(mat, k=5)

    print('-' * 80)
    print(model.knn)
    print('-' * 80)

    model.predict([1])
