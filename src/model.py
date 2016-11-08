import numpy as np


class Model(object):
    def __init__(self, sim_fn, item_based=False):
        self.sim_fn = sim_fn
        self.item_based = item_based


class Simple_sim(Model):
    def train(self, ui_mat):
        if self.item_based:
            ui_mat = ui_mat.T

        # sim_mat
        # shape: (# item, # item) for item based
        # shape: (# user, # user) for user based
        # Note sim_mat is symmetric
        self.sim_mat = self.sim_fn(ui_mat)

    def predict_slow(self, ui_mat, test_mat):
        if self.item_based:
            pred = test_mat.copy()
            for u, i in zip(*pred.nonzero()):
                sub_sim_mat = self.sim_mat[:, i]
                pred[u, i] = ui_mat[u, :].toarray().flatten().dot(sub_sim_mat)
                pred[u, i] /= np.sum(np.abs(sub_sim_mat))
        else:
            pred = test_mat.copy()
            for u, i in zip(*pred.nonzero()):
                sub_sim_mat = self.sim_mat[u, :]
                pred[u, i] = ui_mat[:, i].toarray().flatten().dot(sub_sim_mat)
                pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        return pred[test_mat.nonzero()]

    def predict(self, ui_mat, test_mat):
        if self.item_based:
            rows, cols = test_mat.nonzero()
            U = ui_mat[rows, :]
            S = self.sim_mat[:,cols]
            N = np.sum(S, axis=0)
            p = np.asarray(U.multiply(S.T).sum(axis=1)).flatten() / N
        else:
            rows, cols = test_mat.nonzero()
            U = ui_mat[:, cols]
            S = self.sim_mat[rows, :]
            N = np.sum(S, axis=1)
            p = np.asarray(U.multiply(S.T).sum(axis=0)).flatten() / N
        return p


class Topk(Model):
    def __init__(self, item_based=False, sim_fn=None, k=50):
        self.item_based = item_based
        self.sim_fn = sim_fn
        self.k = k

    def setK(self, k):
        self.k = k

    def train(self, ui_mat):
        if self.item_based:
            ui_mat = ui_mat.T
        self.sim_mat = self.sim_fn(ui_mat)

        self.neighbors = np.zeros((self.sim_mat.shape[0], self.sim_mat.shape[0]), dtype=int)
        self.neighbors = np.argsort(self.sim_mat)[:,::-1]  # do not consider the item itself

    def predict_slow(self, ui_mat, test_mat):

        if self.item_based:
            ui_mat = ui_mat.T
            test_mat = test_mat.T

        pred = test_mat.copy()
        topk_mat = self.neighbors[:, :self.k+1]
        for u, i in zip(*test_mat.nonzero()):
            topk_indices = topk_mat[u]
            sub_sim_mat = self.sim_mat[u, topk_indices].reshape((1,-1))
            pred[u, i] = ui_mat[:, i][topk_indices].dot(sub_sim_mat)
            pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        if self.item_based:
            return pred.T
        else:
            return pred

    def predict(self, ui_mat, test_mat):
        '''Not implemented yet'''
