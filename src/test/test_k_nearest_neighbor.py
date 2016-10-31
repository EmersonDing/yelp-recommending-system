import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from ..k_nearest_neighbor import items_pearson_similarity
from ..k_nearest_neighbor import knn


def test_items_pearson_similarity():
    x = np.array([[1, 3, 5], [4, 2, 6], [3, 4, 2], [6,7,8]])
    assert_allclose(np.corrcoef(x, rowvar=0),
                    items_pearson_similarity(csr_matrix(x)))


def test_knn():
    x = np.array([[1, 3, 5], [1, 3, 4], [4, 2, 1], [4, 2, 2]])
    x = x.T
    neighbors = [[1,3,2], [0,3,2],[3,0,1],[2,0,1]]
    assert_allclose(neighbors,
                    knn(csr_matrix(x), k=3))
