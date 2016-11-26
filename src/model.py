import numpy as np
import math
from scipy import sparse
from spop import spsub, spmul


class Simple_sim(object):
    def __init__(self, sim_fn):
        self.sim_fn = sim_fn

    def train(self, ui_mat, test_mat):
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
        R = ui_mat[:, cols].T
        S = self.sim_mat[rows, :]
        N = np.sum(S, axis=1).flatten()
        return np.asarray(R.multiply(S).sum(axis=1)).flatten() / N


class TopK(object):
    def __init__(self, k, sim_fn):
        self.k = k
        self.sim_fn = sim_fn

    def train(self, ui_mat, test_mat):
        if self.k > ui_mat.shape[0]-1:
            print('==Warning! k is smaller than maximum neighbor number ({0}). Set k to {0}'.format(ui_mat.shape[0]-1))
            self.k = ui_mat.shape[0]-1

        self.sim_mat = self.sim_fn(ui_mat)
        sorted_sim_mat = np.argsort(self.sim_mat)[:,::-1]
        # k nearest neighbor of user u
        u_neighbors = sorted_sim_mat[:, 1:self.k+1]
        u_neighbors.sort()

        rows, cols = test_mat.nonzero()

        Sk_rows = []
        Sk_cols = []
        Sk_data = []
        for ind, (u, i) in enumerate(zip(rows, cols)):
            Sk = list(u_neighbors[u])
            Sk_rows.extend([ind]*len(Sk))
            Sk_cols.extend(Sk)
            Sk_data.extend([1]*len(Sk))

        self.SK = sparse.csr_matrix((Sk_data, (Sk_rows, Sk_cols)), shape=(len(rows), ui_mat.shape[0]))

    def predict(self, ui_mat, test_mat):
        rows, cols = test_mat.nonzero()
        R = ui_mat[:, cols].T
        S = self.sim_mat[rows, :]
        MS = self.SK.multiply(sparse.csr_matrix(S))
        N = np.sum(MS, axis=1)
        return np.asarray(R.multiply(MS).sum(axis=1)).flatten() / N.flatten()


class Model(object):
    def train(self, ui_mat, test_mat):
        ''' Do batch gradient descent to minimize mse error.
        @param ui_mat: 2D sparse matrix. Training matrix.
        '''
        self.init_non_param(ui_mat, test_mat)
        self.init_param(ui_mat)

        for _ in range(self.iteration):
            pred_data = self.predict(ui_mat, ui_mat)
            gradient = self.gradient(ui_mat, pred_data)
            assert len(self.parameters)==len(gradient)
            for param, g in zip(self.parameters, gradient):
                if sparse.issparse(param):
                    param.data += self.gamma * g.data
                else:
                    param += self.gamma * g

    def init_param(self, ui_mat):
        '''Initialize model parameters that are updated in each iteration during training. Should be implemented by derived class.
        @param ui_mat: 2D sparse matrix. Training matrix.
        '''
        raise NotImplementedError

    def init_non_param(self, ui_mat, test_mat):
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

    def init_non_param(self, ui_mat, test_mat):
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
    def __init__(self, sim_fn, k=500, with_c=False, gamma=0.005, lambda4=0.002, iteration=15):
        ''' Neighborhood model, refer to report. Note this implementation use sparse matrix. For a dense counterpart, please refer to obsolete.Neighbor.
        @param k: how many neighbor to use.
        @param with_c: whether to add a bias term c (see report)
        @param gamma: float. learning rate.
        @param lambda4: float. regularition weight.
        @param iteration: sgd iteration.
        '''
        self.sim_fn = sim_fn
        self.k = k
        self.with_c = with_c
        self.gamma = gamma
        self.lambda4 = lambda4
        self.iteration = iteration

    def init_param(self, ui_mat):
        ''' @see Model.init_param.
        '''
        super(Neighbor, self).init_param(ui_mat)
        num_user = ui_mat.shape[0]
        rows = []
        cols = []
        data = [1e-9]*self.k*num_user
        for u in range(num_user):
            rows.extend([u]*(self.k))
            cols.extend(self.u_neighbors[u])

        self.w = sparse.csr_matrix((data, (rows, cols)), shape=(num_user, num_user))
        self.parameters += [self.w]
        if self.with_c:
            self.c = sparse.csr_matrix((data, (rows, cols)), shape=(num_user, num_user))
            self.parameters += [self.c]

    def init_non_param(self, ui_mat, test_mat):
        ''' @see Model.init_non_param.
        '''
        super(Neighbor, self).init_non_param(ui_mat, test_mat)

        if self.k > ui_mat.shape[0]-1:
            print('==Warning! k is smaller than maximum neighbor number ({0}). Set k to {0}'.format(ui_mat.shape[0]-1))
            self.k = ui_mat.shape[0]-1
        self.sim_mat = self.sim_fn(ui_mat)
        sorted_sim_mat = np.argsort(self.sim_mat)[:,::-1]
        # k nearest neighbor of user u
        self.u_neighbors = sorted_sim_mat[:, 1:self.k+1]
        self.u_neighbors.sort()

        uimat_csc = ui_mat.tocsc()
        # user who rates item i
        rated_user = [uimat_csc[:,i].indices for i in range(ui_mat.shape[1])]

        def get_mat_info(mat):
            rows, cols = mat.nonzero()
            N = np.ones(len(rows))

            R_rows = []
            R_cols = []
            R_data = []
            for ind, (u, i) in enumerate(zip(rows, cols)):
                Rkui = set(self.u_neighbors[u]).intersection(set(rated_user[i]))
                Rkui = sorted(list(Rkui))
                if len(Rkui)>0:
                    N[ind] = 1./math.sqrt(len(Rkui))
                R_rows.extend([ind]*len(Rkui))
                R_cols.extend(Rkui)
                R_data.extend([1]*len(Rkui))

            RKUI = sparse.csr_matrix((R_data, (R_rows, R_cols)), shape=(len(rows), mat.shape[0]))
            # RKUI = RKUI.toarray()
            return {'N': N, 'RKUI': RKUI}

        self.mat_info = {id(ui_mat): get_mat_info(ui_mat),
                         id(test_mat): get_mat_info(test_mat)}

    def gradient(self, ui_mat, pred_data):
        ''' @see Model.gradient.
        '''
        rows, cols = ui_mat.nonzero()
        mat_info = self.mat_info[id(ui_mat)]
        N, Rkui = mat_info['N'], mat_info['RKUI']
        R = ui_mat[:, cols].T

        eui = (np.asarray(ui_mat[rows, cols]).flatten() - pred_data)
        w = self.w[rows, :]

        Rmbuj = spsub(R, self.mu)
        Rmbuj = spsub(Rmbuj, self.bu[None, :])
        Rmbuj = spsub(Rmbuj, self.bi[cols][:, None])

        deta = Rkui.multiply(spmul((N*eui)[:, None], Rmbuj)) - spmul(self.lambda4, Rkui.multiply(w))

        gw = self.w.copy()
        gw_data = []

        for u in range(ui_mat.shape[0]):
            tmp = deta[rows==u].sum(axis=0).A1
            tmp = tmp[self.u_neighbors[u]].tolist()
            gw_data.extend(tmp)

        gw.data = np.asarray(gw_data)

        gradient = super(Neighbor, self).gradient(ui_mat, pred_data)
        gradient += [gw]

        if self.with_c:
            detac = (spmul(Rkui, (N*eui)[:, None], copy=True)) - spmul(self.lambda4, Rkui.multiply(w))
            gc = self.c.copy()
            gc_data = []

            for u in range(ui_mat.shape[0]):
                tmp = detac[rows==u].sum(axis=0).A1
                tmp = tmp[self.u_neighbors[u]].tolist()
                gc_data.extend(tmp)

            gc.data = np.asarray(gc_data)
            gradient += [gc]

        return gradient

    def predict(self, ui_mat, test_mat):
        ''' @see Model.predict.
        '''
        pred = super(Neighbor, self).predict(ui_mat, test_mat)
        rows, cols = test_mat.nonzero()

        mat_info = self.mat_info[id(test_mat)]
        N, Rkui = mat_info['N'], mat_info['RKUI']

        w = self.w[rows, :]
        R = ui_mat[:, cols].T
        Rmbuj = spsub(R, self.mu)
        Rmbuj = spsub(Rmbuj, self.bu[None, :])
        Rmbuj = spsub(Rmbuj, self.bi[cols][:, None])

        pred += N * Rkui.multiply(w).multiply(Rmbuj).sum(axis=1).A1
        if self.with_c:
            c = self.c[rows, :]
            pred += N * Rkui.multiply(c).sum(axis=1).A1
        return pred


class Factor(Bias):
    def __init__(self, emb_dim=100, gamma=0.005, lambda4=0.002, iteration=15):
        ''' Factor model, refer to report.
        @param gamma: float. learning rate.
        @param lambda4: float. regularition weight.
        @param iteration: sgd iteration.
        '''
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.lambda4 = lambda4
        self.iteration = iteration

    def init_param(self, ui_mat):
        ''' @see Model.init_param.
        '''
        super(Factor, self).init_param(ui_mat)
        num_user, num_item = ui_mat.shape
        self.P = np.random.random((num_user, self.emb_dim))/2
        self.Q = np.random.random((num_item, self.emb_dim))/2
        self.parameters += [self.P, self.Q]

    def init_non_param(self, ui_mat, test_mat):
        ''' @see Model.init_non_param.
        '''
        super(Factor, self).init_non_param(ui_mat, test_mat)

    def gradient(self, ui_mat, pred_data):
        ''' @see Model.gradient.
        '''

        rows, cols = ui_mat.nonzero()
        eui = (np.asarray(ui_mat[rows, cols]).flatten() - pred_data)[:, None]

        P = self.P[rows, :]
        Q = self.Q[cols, :]

        detaP = eui * Q - self.lambda4 * P
        detaQ = eui * P - self.lambda4 * Q

        gradient = super(Factor, self).gradient(ui_mat, pred_data)

        gp = np.zeros_like(self.P)
        gq = np.zeros_like(self.Q)

        for ind, (u, i) in enumerate(zip(rows, cols)):
            gp[u] += detaP[ind]
            gq[i] += detaQ[ind]

        gradient = super(Factor, self).gradient(ui_mat, pred_data)
        gradient += [gp, gq]

        return gradient

    def predict(self, ui_mat, test_mat):
        ''' @see Model.predict.
        '''
        pred = super(Factor, self).predict(ui_mat, test_mat)
        rows, cols = test_mat.nonzero()

        P = self.P[rows, :]
        Q = self.Q[cols, :]
        pred += np.sum(P*Q, axis=1)/self.emb_dim
        return pred
