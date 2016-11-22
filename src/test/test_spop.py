import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_allclose
from ..spop import spadd, spsub, spmul


a = np.eye(3)
a[0,1] = 1
a=csr_matrix(a)
value = 2
values = np.array([1,2,3])
vec = np.array([1,2,3])


def test_spadd():
    assert_allclose(spadd(a, value, copy=True).data, [3,3,3,3])
    assert_allclose(spadd(a, values[:, None], copy=True).data, [2,2,3,4])
    assert_allclose(spadd(a, values[None, :], copy=True).data, [2,3,3,4])


def test_spsub():
    assert_allclose(spsub(a, value, copy=True).data, [-1,-1,-1,-1])
    assert_allclose(spsub(a, values[:, None], copy=True).data, [0,0,-1,-2])
    assert_allclose(spsub(a, values[None, :], copy=True).data, [0,-1,-1,-2])


def test_spmul():
    assert_allclose(spmul(a, value, copy=True).data, [2,2,2,2])
    assert_allclose(spmul(a, values[:, None], copy=True).data, [1,1,2,3])
    assert_allclose(spmul(a, values[None, :], copy=True).data, [1,2,2,3])
