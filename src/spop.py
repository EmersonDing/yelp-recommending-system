import numpy as np
from scipy import sparse
import operator


def determine_sp_dense(spmat, vec):
    if sparse.issparse(spmat):
        return spmat, vec
    else:
        return vec, spmat


def spop(spmat, array, op, copy):
    spmat, array = determine_sp_dense(spmat, array)
    # make sure we are dealing with csr_matrix
    spmat = spmat.tocsr()
    if not hasattr(array, 'shape') or len(array.shape)<1:
        # case 2, array is a single value
        res_data = op(spmat.data, array)
    else:
        assert len(array.shape)==2
        if array.shape[0]==1:
            # case 3, array is a row vector, add array to each row
            assert array.shape[1]==spmat.shape[1]
            res_data = op(spmat.data, array[0, spmat.indices])
        else:
            # case 4, array is a col vector, add each element to each row
            assert array.shape[0]==spmat.shape[0]
            nnz_per_row = np.diff(spmat.indptr)
            res_data = op(spmat.data, np.repeat(array[:,0], nnz_per_row))

    if copy:
        res = spmat.copy()
    else:
        res = spmat

    res.data = res_data
    return res


def spadd(spmat, values, copy=False):
    return spop(spmat, values, operator.add, copy)


def spsub(spmat, values, copy=False):
    return spop(spmat, values, operator.sub, copy)


def spmul(spmat, values, copy=False):
    return spop(spmat, values, operator.mul, copy)
