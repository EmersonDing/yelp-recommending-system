import numpy as np
import math
from model import Bias


class Neighbor(Bias):
    def __init__(self, sim_fn, k=500, gamma=0.005, lambda4=0.002, iteration=15):
        ''' Obsolete Neighborhood model, refer to report. This implementation use dense vector, which leads to memory crach when number of user is too large.
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
        u_neighbors = sorted_sim_mat[:, 1:self.k+1]
        # user who rates item i
        rated_user = [ui_mat.tocsc()[:,i].indices for i in range(ui_mat.shape[1])]

        def get_mat_info(mat):
            rows, cols = mat.nonzero()
            N = np.ones(len(rows))
            RKUI= np.zeros((len(rows), mat.shape[0]))

            for ind, (u, i) in enumerate(zip(rows, cols)):
                Rkui = set(u_neighbors[u]).intersection(set(rated_user[i]))
                Rkui = sorted(list(Rkui))
                if len(Rkui)>0:
                    N[ind] = 1./math.sqrt(len(Rkui))
                RKUI[ind, Rkui] = 1
            return {'N': N, 'RKUI': RKUI}

        self.mat_info = {id(ui_mat): get_mat_info(ui_mat),
                         id(test_mat): get_mat_info(test_mat)}

    def gradient(self, ui_mat, pred_data):
        ''' @see Model.gradient.
        '''
        rows, cols = ui_mat.nonzero()
        mat_info = self.mat_info[id(ui_mat)]
        N, Rkui = mat_info['N'], mat_info['RKUI']
        R = ui_mat[:, cols].T.toarray()

        eui = (np.asarray(ui_mat[rows, cols]).flatten() - pred_data)[:, None]
        w = self.w[rows, :]
        Rmbuj = R-self.mu
        Rmbuj = Rmbuj - self.bu[None, :]
        Rmbuj = Rmbuj - self.bi[cols][:, None]

        deta = Rkui * (N[:, None]* eui * Rmbuj - self.lambda4 * w)

        gw = np.zeros_like(self.w)

        for u in range(ui_mat.shape[0]):
            gw[u] = deta[rows==u].sum(axis=0)

        gradient = super(Neighbor, self).gradient(ui_mat, pred_data)
        gradient += [gw]

        return gradient

    def gradient_slow(self, ui_mat):
        gradient = super(Neighbor, self).gradient_slow(ui_mat)

        rows, cols = ui_mat.nonzero()

        pred_data = self.predict(ui_mat, ui_mat)
        pred_csr = sparse.csr_matrix((pred_data, (rows, cols)))

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

    def predict_slow(self, ui_mat, test_mat):
        rows, cols = test_mat.nonzero()
        mat_info = self.mat_info[id(test_mat)]
        N, Rkui = mat_info['N'], mat_info['RKUI']
        Rkui = Rkui.astype(bool)

        pred = super(Neighbor, self).predict_slow(ui_mat, test_mat)
        for ind, (u, i) in enumerate(zip(rows, cols)):
            rkui = Rkui[ind]
            w = self.w[u, rkui]
            r = ui_mat[rkui, i].toarray().flatten()
            buj = self.mu+self.bu[rkui]+self.bi[i]
            pred[ind] += N[ind] * w.dot(r - buj)
        return pred

    def predict(self, ui_mat, test_mat):
        ''' @see Model.predict.
        '''
        pred = super(Neighbor, self).predict(ui_mat, test_mat)
        rows, cols = test_mat.nonzero()

        mat_info = self.mat_info[id(test_mat)]
        N, Rkui = mat_info['N'], mat_info['RKUI']

        w = self.w[rows, :]
        R = ui_mat[:, cols].T.toarray()
        Rmbuj = R-self.mu
        Rmbuj = Rmbuj - self.bu[None, :]
        Rmbuj = Rmbuj - self.bi[cols][:, None]
        pred += N * (Rkui * w * (Rmbuj)).sum(axis=1)
        return pred
