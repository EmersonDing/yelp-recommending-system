import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from ..k_nearest_neighbor import items_pearson_similarity
from ..k_nearest_neighbor import find_item_neighbors


def test_items_pearson_similarity():
    x = np.array([[1, 3, 5, 0], [4, 2, 2, 0], [3, 4, 3, 0], [6,7,0, 0]], dtype=float)
    # Note that the fourth column is all zero, leading to nan in the coefficient matrix, we set nan to 0 (non-correlated) as we did in items_pearson_similarity
    npres = np.corrcoef(x, rowvar=0)
    npres[np.isnan(npres)] = 0
    assert_allclose(npres,
                    items_pearson_similarity(csr_matrix(x)))


def test_find_item_neighbors():
    x = np.array([[1, 3, 5], [1, 3, 4], [4, 2, 1], [4, 2, 2]])
    x = x.T
    neighbors = [[1,3,2], [0,3,2],[3,0,1],[2,0,1]]
    assert_allclose(neighbors,
                    find_item_neighbors(csr_matrix(x), k=3))
