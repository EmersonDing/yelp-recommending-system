import numpy as np


class Simple_sim(object):
    def __init__(self, sim_fn, **kwargs):
        self.sim_fn = sim_fn

    def train(self, ui_mat):
        self.sim_mat = self.sim_fn(ui_mat)

    def predict(self, ui_mat, test_mat):
        pred = test_mat.copy()
        for u, i in zip(*pred.nonzero()):
            sub_sim_mat = self.sim_mat[u, :]
            pred[u, i] = ui_mat[:, i].flatten().dot(sub_sim_mat)
            pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        return pred[test_mat.nonzero()]


class Topk(object):
    def __init__(self, sim_fn=None, k=50):
        self.sim_fn = sim_fn
        self.k = k

    def setK(self, k):
        self.k = k

    def train(self, ui_mat):
        self.sim_mat = self.sim_fn(ui_mat)

        self.neighbors = np.zeros((self.sim_mat.shape[0], self.sim_mat.shape[0]), dtype=int)
        self.neighbors = np.argsort(self.sim_mat)[:,::-1]

    def predict(self, ui_mat, test_mat):
        pred = test_mat.copy()
        topk_mat = self.neighbors[:, :self.k+1]
        for u, i in zip(*test_mat.nonzero()):
            topk_indices = topk_mat[u]
            sub_sim_mat = self.sim_mat[u, topk_indices]
            pred[u, i] = ui_mat[:, i][topk_indices].dot(sub_sim_mat)
            pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        return pred[test_mat.nonzero()]
