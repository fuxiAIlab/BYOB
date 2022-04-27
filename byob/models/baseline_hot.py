import os
import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


class HotModel(object):
    """
    Hot Model for Sequential Recommendation
    """

    def __init__(self, config):
        super(HotModel, self).__init__()

        self.model_type = 'HOT'

        self.k = config['rank_k']
        self.shuffle = config['shuffle']

        self.mat = None
        self.hot_items = None
        self.hot_bundles = None

    def fit(self, mat, k=5):
        """
        https://stackoverflow.com/questions/3337301/numpy-matrix-to-array
        https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        """
        if k is None:
            k = self.k
        self.mat = mat
        vec = self.mat.sum(axis=0)  # count the users by items
        vec = vec.A1  # return self as a flattened ndarray
        rank_k = vec.argsort()[-k:][::-1]
        self.hot_items = rank_k
        return rank_k

    def apriori(self, mat, support=0.5, k=5):
        """
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

        """
        self.mat = mat
        print(mat.shape, mat.nnz)
        n_user, n_item = mat.shape

        dataset = []
        array = mat.toarray()[:, 0:1000]
        # array = mat.todense().A
        # print(array.shape)
        for i in range(array.shape[0]):
            transaction = []
            for j in range(array.shape[1]):
                if array[i][j] != 0:
                    transaction.append(j)
            dataset.append(transaction)

        te = TransactionEncoder()

        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # print(te_ary.shape, df.shape)

        df = apriori(df, min_support=support, use_colnames=True)
        # print(df.shape, df.columns)

        df['length'] = df['itemsets'].apply(lambda x: len(x))
        df = df[(df['length'] == k) & (df['support'] >= support)]
        print(df.head())

        bundles = []
        for support, itemset, length in df.sort_values(by='support', ascending=False).values:
            bundle = []
            for i in itemset:
                bundle.append(i)
            bundles.append(bundle)
        self.hot_bundles = bundles

        return bundles

    def predict(self, u, i):
        return 1.0

    def recommend(self, u, i, k=5):
        assert k <= len(self.hot_items)
        # np.random.choice(self.hot_items, k, replace=False)
        hot_items = np.random.permutation(self.hot_items)  # random shuffle
        top_k = hot_items[:k]
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

    model = HotModel()

    model.fit(mat, k=5)

    print('-' * 80)
    print(model.hot_items)
    print('-' * 80)

    model.apriori(mat, support=0.1, k=5)

    print('-' * 80)
    print(model.hot_bundles)
    print('-' * 80)

    model.predict([1])
