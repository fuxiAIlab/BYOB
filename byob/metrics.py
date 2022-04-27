import numpy as np

from byob.utils import to_array, to_tensor


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def ordinary_accuracy(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    count = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return count / total


def binary_accuracy(y_true, y_pred, threshold=0.5, sigmoid=True):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape[0] == y_pred.shape[0]
    if sigmoid:
        y_pred = _sigmoid(y_pred)
    # y_pred = np.round(y_pred)
    # y_pred = np.greater(y_pred, threshold)
    y_pred = (y_pred > threshold).astype(int)
    count = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return count / total


def categorical_accuracy(y_true, y_pred, softmax=False):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape[0] == y_pred.shape[0]
    if softmax:
        y_pred = _softmax(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    count = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return count / total


def bundle_precision(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    # a = (y_pred[:,0:1] == y_true).sum(axis=1)
    # precision = np.count_nonzero(a) / len(y_true)
    precision = 0.0
    common = []
    for i in range(len(y_true)):
        a = np.intersect1d(y_true[i], y_pred[i])
        precision += float(y_pred[i][0] in a)
        common.append(len(a))
    precision /= len(y_true)
    # print(common)
    return precision


def bundle_precision_plus(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    precision_plus = 0.0
    common = []
    for i in range(len(y_true)):
        a = np.intersect1d(y_true[i], y_pred[i])
        precision_plus += float(len(a) > 0)
        common.append(len(a))
    precision_plus /= len(y_true)
    # print(common)
    return precision_plus


def bundle_recall(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    recall = 0.0
    common = []
    for i in range(len(y_true)):
        a = np.intersect1d(y_true[i], y_pred[i])
        recall += len(a) / len(y_true[0])
        common.append(len(a))
    recall /= len(y_true)
    # print(common)
    return recall


def bundle_metrics(y_true, y_pred):
    # print(y_true.shape, y_true.dtype, y_true[0])
    # print(y_pred.shape, y_pred.dtype, y_pred[0])
    precision = bundle_precision(y_true, y_pred)
    precision_plus = bundle_precision_plus(y_true, y_pred)
    recall = bundle_recall(y_true, y_pred)
    return {'precision': precision,
            'precision_plus': precision_plus,
            'recall': recall}


if __name__ == "__main__":
    # y_true = np.random.randint(30, size=(100, 5))
    # y_pred = np.random.randint(30, size=(100, 5))
    # print(bundle_precision(y_true, y_pred))
    # print(bundle_precision_plus(y_true, y_pred))
    # print(bundle_recall(y_true, y_pred))
    pass
