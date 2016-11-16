import numpy as np
import math
from scipy import sparse


class Simple_sim(object):
    def __init__(self, sim_fn):
        self.sim_fn = sim_fn

    def train(self, ui_mat):
        # sim_mat
        # shape: (# item, # item) for item based
        # shape: (# user, # user) for user based
        # Note sim_mat is symmetric
        self.sim_mat = self.sim_fn(ui_mat)

    def predict_slow(self, ui_mat, test_mat):
        pred = test_mat.copy()
        for u, i in zip(*pred.nonzero()):
            sub_sim_mat = self.sim_mat[u, :]
            pred[u, i] = ui_mat[:, i].toarray().flatten().dot(sub_sim_mat)
            pred[u, i] /= np.sum(np.abs(sub_sim_mat))

        return pred[test_mat.nonzero()]

    def predict(self, ui_mat, test_mat):
        rows, cols = test_mat.nonzero()
        U = ui_mat[:, cols]
        S = self.sim_mat[rows, :]
        N = np.sum(S, axis=1)
        return np.asarray(U.multiply(S.T).sum(axis=0)).flatten() / N


class Bias(object):
    def __init__(self, gamma=0.005, lambda4=0.002, iteration=15):
        self.gamma = gamma
        self.lambda4 = lambda4
        self.iteration = iteration

    def train(self, ui_mat):
        gamma, lambda4 = self.gamma, self.lambda4  # just to make it short
        self.mu = ui_mat.sum()/ui_mat.getnnz()
        self.bu = np.zeros(ui_mat.shape[0])
        self.bi = np.zeros(ui_mat.shape[1])

        rows, cols = ui_mat.nonzero()
        user_nnz = ui_mat.getnnz(axis=1)
        item_nnz = ui_mat.getnnz(axis=0)
        user_sum = ui_mat.sum(axis=1).A1
        item_sum = ui_mat.sum(axis=0).A1

        for _ in range(self.iteration):
            pred_data = self.predict(ui_mat, ui_mat)
            pred_coo = sparse.coo_matrix((pred_data, (rows, cols)))
            pred_user_sum = pred_coo.tocsr().sum(axis=1).A1
            pred_item_sum = pred_coo.tocsc().sum(axis=0).A1
            self.bu += gamma * (user_sum-pred_user_sum) - gamma * lambda4 * user_nnz * self.bu
            self.bi += gamma * (item_sum-pred_item_sum) - gamma * lambda4 * item_nnz * self.bi

    def train_slow(self, ui_mat, gamma=0.005, lambda4=0.002, iteration=15):
        self.mu = ui_mat.sum()/ui_mat.getnnz()
        self.bu = np.zeros(ui_mat.shape[0])
        self.bi = np.zeros(ui_mat.shape[1])

        rows, cols = ui_mat.nonzero()
        for _ in range(iteration):
            pred_data = self.predict(ui_mat, ui_mat)
            pred_csr = sparse.csr_matrix((pred_data, (rows, cols)))

            deta_bu = np.zeros_like(self.bu)
            deta_bi = np.zeros_like(self.bi)
            for u, i in zip(rows, cols):
                eui = ui_mat[u,i] - pred_csr[u,i]
                deta_bu[u] += (eui - lambda4* self.bu[u])
                deta_bi[i] += (eui - lambda4* self.bi[i])

            self.bu += gamma * deta_bu
            self.bi += gamma * deta_bi

    def predict(self, ui_mat, test_mat):
        rows, cols = test_mat.nonzero()
        p = self.bu[rows] + self.bi[cols] + self.mu
        return p


class Neighbor(object):
    def __init__(self, sim_fn, k=500, gamma=0.005, lambda4=0.002, iteration=15, **kwargs):
        super(Neighbor, self).__init__(**kwargs)
        self.sim_fn = sim_fn
        self.k = k
        self.gamma = gamma
        self.lambda4 = lambda4
        self.iteration = iteration

    def train(self, ui_mat):
        gamma, lambda4 = self.gamma, self.lambda4  # just to make it short
        k = min(self.k, ui_mat.shape[0])
        ui_mat_csc = ui_mat.tocsc()

        self.mu = ui_mat.sum()/ui_mat.getnnz()
        self.bu = np.zeros(ui_mat.shape[0])
        self.bi = np.zeros(ui_mat.shape[1])

        self.sim_mat = self.sim_fn(ui_mat)
        sorted_sim_mat = np.argsort(self.sim_mat)[:,::-1]
        # k nearest neighbor of user u
        self.u_neighbors = sorted_sim_mat[:, 1:k+1]
        # user who rates item i
        self.rated_user = [ui_mat_csc[:,i].indices for i in range(ui_mat_csc.shape[1])]
        self.w = np.zeros_like(self.sim_mat)  # TODO, could use less memory
        # cahche of Rkui, user u's neighbor for item i (intersection of k nearest neighbor of user u and users that rated item i)
        self.cached_Rkui = {}

        rows, cols = ui_mat.nonzero()
        # number of items user u rated
        user_nnz = ui_mat.getnnz(axis=1)
        # number of users that rated item i
        item_nnz = ui_mat.getnnz(axis=0)
        # user u's rating sum
        user_sum = ui_mat.sum(axis=1).A1
        # item i's rating sum
        item_sum = ui_mat.sum(axis=0).A1

        for _ in range(self.iteration):
            # do bias model
            pred_data = self.predict(ui_mat, ui_mat)
            pred_coo = sparse.coo_matrix((pred_data, (rows, cols)))
            pred_csr = pred_coo.tocsr()
            pred_csc = pred_coo.tocsc()

            pred_user_sum = pred_csr.sum(axis=1).A1
            pred_item_sum = pred_csc.sum(axis=0).A1
            deta_bu = gamma * (user_sum-pred_user_sum) - gamma * lambda4 * user_nnz * self.bu
            deta_bi = gamma * (item_sum-pred_item_sum) - gamma * lambda4 * item_nnz * self.bi

            # do neighbor model
            for u, i in zip(rows, cols):
                eui = ui_mat[u,i] - pred_csr[u, i]
                Rkui = self.get_Rkui(u, i)
                if len(Rkui)>0:
                    deta = 1./math.sqrt(len(Rkui)) * eui
                    deta *= (ui_mat[Rkui,i].toarray().flatten()-(self.mu+self.bu[Rkui]+self.bi[i]))
                    deta -= lambda4 * self.w[u, Rkui]
                    self.w[u, Rkui] += gamma * deta

            self.bu += deta_bu
            self.bi += deta_bi

    def get_Rkui(self, u, i):
        if (u,i) not in self.cached_Rkui:
            Rkui = set(self.u_neighbors[u]).intersection(set(self.rated_user[i]))
            Rkui = sorted(list(Rkui))
            self.cached_Rkui[(u,i)] = Rkui
            return Rkui
        else:
            return self.cached_Rkui[(u,i)]

    def predict(self, ui_mat, test_mat):
        rows, cols = test_mat.nonzero()
        bui = self.bu[rows] + self.bi[cols] + self.mu
        pred_data = bui
        for ind, (u, i) in enumerate(zip(rows, cols)):
            Rkui = self.get_Rkui(u, i)
            if len(Rkui)>0:
                pred_data[ind] += 1./math.sqrt(len(Rkui)) * self.w[u, Rkui].dot(ui_mat[Rkui, i].toarray().flatten() - (self.mu+self.bu[Rkui]+self.bi[i]))

        return pred_data
