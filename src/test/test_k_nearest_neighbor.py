import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from ..k_nearest_neighbor import column_covariance, items_pearson_similarity


def test_items_pearson_similarity():
    x = np.array([[1, 3, 5], [4, 2, 6], [3, 4, 2], [6,7,8]])
    assert_allclose(np.corrcoef(x, rowvar=0),
                    items_pearson_similarity(csr_matrix(x)))


def test_column_covariance():
    x = np.array([[1, 3, 5], [4, 2, 6], [3, 4, 2], [6,7,8]])
    assert_allclose(np.cov(x, rowvar=0),
                    column_covariance(csr_matrix(x)))
