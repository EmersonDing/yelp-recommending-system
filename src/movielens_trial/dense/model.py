import numpy as np


class Model(object):
    def __init__(self, sim_fn, item_based=False):
        self.sim_fn = sim_fn
        self.item_based = item_based


class Simple_sim(Model):
    def train(self, ui_mat):
        if self.item_based:
            ui_mat = ui_mat.T
        self.sim_mat = self.sim_fn(ui_mat)
        # print avg(self.sim_mat)

    def predict(self, ui_mat, test_mat):
        if self.item_based:
            pred = test_mat.copy()
            for u, i in zip(*pred.nonzero()):
                sub_sim_mat = self.sim_mat[:, i]
                pred[u, i] = ui_mat[u, :].flatten().dot(sub_sim_mat)
                pred[u, i] /= np.sum(np.abs(sub_sim_mat))
        else:
            pred = test_mat.copy()
            for u, i in zip(*pred.nonzero()):
                sub_sim_mat = self.sim_mat[u, :]
                pred[u, i] = ui_mat[:, i].flatten().dot(sub_sim_mat)
                pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        return pred[test_mat.nonzero()]


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
        self.neighbors = np.argsort(self.sim_mat)[:,::-1]

    def predict(self, ui_mat, test_mat):

        if self.item_based:
            ui_mat = ui_mat.T
            test_mat = test_mat.T

        pred = test_mat.copy()
        topk_mat = self.neighbors[:, :self.k+1]
        for u, i in zip(*test_mat.nonzero()):
            topk_indices = topk_mat[u]
            sub_sim_mat = self.sim_mat[u, topk_indices]
            pred[u, i] = ui_mat[:, i][topk_indices].dot(sub_sim_mat)
            pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        return pred[test_mat.nonzero()]
