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


class Model(object):
    def train(self, ui_mat):
        ''' Do batch gradient descent to minimize mse error.
        @param ui_mat: 2D sparse matrix. Training matrix.
        '''
        self.init_non_param(ui_mat)
        self.init_param(ui_mat)

        for _ in range(self.iteration):
            pred_data = self.predict(ui_mat, ui_mat)
            gradient = self.gradient(ui_mat, pred_data)
            for param, g in zip(self.parameters, gradient):
                param += self.gamma * g

    def init_param(self, ui_mat):
        '''Initialize model parameters that are updated in each iteration during training. Should be implemented by derived class.
        @param ui_mat: 2D sparse matrix. Training matrix.
        '''
        raise NotImplementedError

    def init_non_param(self, ui_mat):
        '''Initialize model parameters that are not updated in each iteration during training (but you don't want to recompute them every iteration). Should be implemented by derived class.
        @param ui_mat: 2D sparse matrix. Training matrix.
        '''
        raise NotImplementedError

    def gradient(self, ui_mat, pred_data):
        '''Calculate the gradient (partial derivative of error function w.r.t model parameters).
        @param ui_mat: 2D sparse matrix. Training matrix.
        @param pred_data: 1D array. The predicted values of each non zero element of ui_mat. Note you could reconstruct the prediction sparse matrix like: `pred_mat = csr_matrix((pred_data, ui_mat.nonzero())`.
        '''

    def predict(self, ui_mat, test_mat):
        '''Predict the ratings for the nonzero entry in test_mat. ui_mat is provided in case we need it.
        @param ui_mat: 2D sparse matrix. Training matrix.
        @param test_mat: 2D sparse matrix. Testing matrix.
        @return list. The length of the return list will be of length test_mat.getnnz()
        '''
        raise NotImplementedError


class Bias(Model):
    def __init__(self, gamma=0.005, lambda4=0.002, iteration=15):
        ''' Bias model, refer to report.
        @param gamma: float. learning rate.
        @param lambda4: float. regularition weight.
        @param iteration: sgd iteration.
        '''
        self.gamma = gamma
        self.lambda4 = lambda4
        self.iteration = iteration

    def init_non_param(self, ui_mat):
        ''' @see Model.init_non_param.
        '''
        self.mu = ui_mat.sum()/ui_mat.getnnz()

    def init_param(self, ui_mat):
        ''' @see Model.init_param.
        '''
        self.bu = np.zeros(ui_mat.shape[0])
        self.bi = np.zeros(ui_mat.shape[1])
        self.parameters = [self.bu, self.bi]
        return self.parameters

    def gradient(self, ui_mat, pred_data):
        ''' @see Model.gradient.
        '''
        rows, cols = ui_mat.nonzero()

        user_nnz = ui_mat.getnnz(axis=1)
        item_nnz = ui_mat.getnnz(axis=0)
        user_sum = ui_mat.sum(axis=1).A1
        item_sum = ui_mat.sum(axis=0).A1

        pred_coo = sparse.coo_matrix((pred_data, (rows, cols)))
        pred_user_sum = pred_coo.tocsr().sum(axis=1).A1
        pred_item_sum = pred_coo.tocsc().sum(axis=0).A1

        return [(user_sum-pred_user_sum) - self.lambda4 * user_nnz * self.bu,
                (item_sum-pred_item_sum) - self.lambda4 * item_nnz * self.bi]

    def gradient_slow(self, ui_mat):
        rows, cols = ui_mat.nonzero()
        pred_data = self.predict(ui_mat, ui_mat)
        pred_csr = sparse.csr_matrix((pred_data, (rows, cols)))

        deta_bu = np.zeros_like(self.bu)
        deta_bi = np.zeros_like(self.bi)
        for u, i in zip(rows, cols):
            eui = ui_mat[u,i] - pred_csr[u,i]
            deta_bu[u] += (eui - self.lambda4 * self.bu[u])
            deta_bi[i] += (eui - self.lambda4 * self.bi[i])
        return [deta_bu, deta_bi]

    def predict(self, ui_mat, test_mat):
        ''' @see Model.predict.
        '''
        rows, cols = test_mat.nonzero()
        p = self.bu[rows] + self.bi[cols] + self.mu
        return p


class Neighbor(Bias):
    def __init__(self, sim_fn, k=500, gamma=0.005, lambda4=0.002, iteration=15):
        ''' Neighborhood model, refer to report.
        @param k: how many neighbor to use.
        @param gamma: float. learning rate.
        @param lambda4: float. regularition weight.
        @param iteration: sgd iteration.
        '''
        self.sim_fn = sim_fn
        self.k = k
        self.gamma = gamma
        self.lambda4 = lambda4
        self.iteration = iteration

    def init_param(self, ui_mat):
        ''' @see Model.init_param.
        '''
        super(Neighbor, self).init_param(ui_mat)
        self.w = np.zeros((ui_mat.shape[0], ui_mat.shape[0]))  # TODO, could use less memory
        self.parameters += [self.w]

    def init_non_param(self, ui_mat):
        ''' @see Model.init_non_param.
        '''
        super(Neighbor, self).init_non_param(ui_mat)

        k = min(self.k, ui_mat.shape[0])
        self.sim_mat = self.sim_fn(ui_mat)
        sorted_sim_mat = np.argsort(self.sim_mat)[:,::-1]
        # k nearest neighbor of user u
        self.u_neighbors = sorted_sim_mat[:, 1:k+1]
        # user who rates item i
        self.rated_user = [ui_mat.tocsc()[:,i].indices for i in range(ui_mat.shape[1])]
        # cahche of Rkui, user u's neighbor for item i (intersection of k nearest neighbor of user u and users that rated item i)
        self.cached_Rkui = {}

    def gradient(self, ui_mat, pred_data):
        ''' @see Model.gradient.
        '''
        rows, cols = ui_mat.nonzero()

        N = np.ones(len(rows))
        eui = (np.asarray(ui_mat[rows, cols]).flatten() - pred_data)[:, None]
        mask = np.zeros((len(rows), ui_mat.shape[0]))
        w = self.w[rows, :]
        R = ui_mat[:, cols].T.toarray()
        bu = np.repeat(self.bu[None, :], axis=0, repeats=len(rows))
        bi = self.bi[cols][:, None]

        for ind, (u, i) in enumerate(zip(rows, cols)):
            Rkui = self.get_Rkui(u, i)
            if len(Rkui)>0:
                N[ind] = 1./math.sqrt(len(Rkui))
            mask[ind, Rkui] = 1

        deta = mask * (N[:, None]* eui * (R - (self.mu + bu+bi)) - self.lambda4 * w)
        gw = np.zeros_like(self.w)

        for ind, u in enumerate(rows):
            gw[u] += deta[ind]

        gradient = super(Neighbor, self).gradient(ui_mat, pred_data)
        gradient += [gw]

        return gradient

    def gradient_slow(self, ui_mat):
        gradient = super(Neighbor, self).gradient_slow(ui_mat)

        rows, cols = ui_mat.nonzero()

        pred_data = self.predict(ui_mat, ui_mat)
        pred_csr = sparse.csr_matrix((pred_data, (rows, cols)))
        # do neighbor model

        gw = np.zeros_like(self.w)
        for u, i in zip(rows, cols):
            eui = ui_mat[u,i] - pred_csr[u, i]
            Rkui = self.get_Rkui(u, i)
            if len(Rkui)>0:
                deta = 1./math.sqrt(len(Rkui)) * eui
                deta *= (ui_mat[Rkui,i].toarray().flatten()-(self.mu+self.bu[Rkui]+self.bi[i]))
                deta -= self.lambda4 * self.w[u, Rkui]
                gw[u, Rkui] += deta

        gradient += [gw]
        return gradient

    def get_Rkui(self, u, i):
        ''' Get the intersection of (1) k most similar user to u and (2) user who rated item i. See report for more information.
        @param u: user id.
        @param i: item id.
        @return list.
        '''
        if (u,i) not in self.cached_Rkui:
            Rkui = set(self.u_neighbors[u]).intersection(set(self.rated_user[i]))
            Rkui = sorted(list(Rkui))
            self.cached_Rkui[(u,i)] = Rkui
            return Rkui
        else:
            return self.cached_Rkui[(u,i)]

    def predict_slow(self, ui_mat, test_mat):
        rows, cols = test_mat.nonzero()
        bui = self.bu[rows] + self.bi[cols] + self.mu
        pred = bui
        for ind, (u, i) in enumerate(zip(rows, cols)):
            Rkui = self.get_Rkui(u, i)
            if len(Rkui)>0:
                pred[ind] += 1./math.sqrt(len(Rkui)) * self.w[u, Rkui].dot(ui_mat[Rkui, i].toarray().flatten() - (self.mu+self.bu[Rkui]+self.bi[i]))

        return pred

    def predict(self, ui_mat, test_mat):
        ''' @see Model.predict.
        '''
        pred = super(Neighbor, self).predict(ui_mat, test_mat)
        rows, cols = test_mat.nonzero()

        N = np.ones(len(rows))
        mask = np.zeros((len(rows), ui_mat.shape[0]))
        w = self.w[rows, :]
        R = ui_mat[:, cols].T.toarray()
        bu = np.repeat(self.bu[None, :], axis=0, repeats=len(rows))
        bi = self.bi[cols][:, None]

        for ind, (u, i) in enumerate(zip(rows, cols)):
            Rkui = self.get_Rkui(u, i)
            if len(Rkui)>0:
                N[ind] = 1./math.sqrt(len(Rkui))
            mask[ind, Rkui] = 1

        tmp = mask * w * (R - (self.mu + bu+bi))
        pred += N * tmp.sum(axis=1)
        return pred
