import numpy as np
import pandas as pd
import similarity
from model import Simple_sim, Topk


def read_data(fname):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(fname, sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ui_mat = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ui_mat[row[1]-1, row[2]-1] = row[3]
    return ui_mat


def train_test_split(ui_mat, split=0.1):
    train_mat = ui_mat.copy()
    test_mat = ui_mat.copy()
    for i in range(ui_mat.shape[0]):
        cols = ui_mat[i].nonzero()[0]
        cols = np.random.permutation(cols)
        cut = int(split*len(cols))
        test_ind = cols[:cut]
        train_ind = cols[cut:]
        test_mat[i, train_ind] = 0
        train_mat[i, test_ind] = 0

    # Test and training are truly disjoint
    assert(np.all((train_mat * test_mat) == 0))
    return train_mat, test_mat

    # different version using random.choice
    test = np.zeros(ui_mat.shape)
    train = ui_mat.copy()
    for user in xrange(ui_mat.shape[0]):
        test_ratings = np.random.choice(ui_mat[user, :].nonzero()[0], size=10, replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ui_mat[user, test_ratings]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test


def get_mse(pred, actual):
    pred = pred.flatten()
    actual = actual[actual.nonzero()].flatten()
    assert(len(pred)==len(actual))
    return np.sum((pred-actual)**2)/len(pred)


def doIt(modelCls, **model_args):
    user_model = modelCls(item_based=False, **model_args)
    item_model = modelCls(item_based=True, **model_args)

    user_model.train(train_mat)
    item_model.train(train_mat)

    user_prediction = user_model.predict(train_mat, test_mat)
    item_prediction = item_model.predict(train_mat, test_mat)
    print
    print('='*20)
    print('Model: {}'.format(modelCls.__name__))
    print('Args: {}'.format(model_args))
    print('User-based CF MSE: {}'.format(get_mse(user_prediction, test_mat)))
    print('Item-based CF MSE: {}'.format(get_mse(item_prediction, test_mat)))

if __name__ == '__main__':
    np.random.seed(0)
    ui_mat = read_data('../u.data')
    train_mat, test_mat = train_test_split(ui_mat, 0.1)

    # Note: simple_sim is just TopK with k=\infty
    doIt(Simple_sim, sim_fn=similarity.cosine_sim)
    doIt(Topk, k=50, sim_fn=similarity.cosine_sim)
