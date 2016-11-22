import numpy as np
from numpy.testing import assert_allclose
from ..similarity import cosine_sim, pearson_sim, shrunk_sim
from scipy.sparse import csr_matrix


def test_cosine_sim():
    x = np.array([[1,0,2], [4,5,0]])
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    npres = np.ones((2,2))
    npres[0,1] = npres[1,0] = np.dot(x[0], x[1])
    assert_allclose(npres,
                    cosine_sim(csr_matrix(x)))


def test_pearson_similarity():
    x = np.array([[1,0,2], [4,5,0]])
    npres = np.corrcoef(x)
    assert_allclose(npres,
                    pearson_sim(csr_matrix(x)))


def test_shrunk_sim():
    x = np.array([[1,0,2], [4,5,0]])
    lambda2 = 100.0
    sim_mat = np.corrcoef(x)
    npres = sim_mat.copy()
    npres[0,1] *= (1/(1+lambda2))
    npres[1,0] *= (1/(1+lambda2))
    assert_allclose(npres,
                    shrunk_sim(csr_matrix(x), sim_mat, lambda2))
