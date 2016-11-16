import numpy as np


def cosine_sim(x, epsilon=1e-9):
    '''
    @param x: 2D (sparse) array. Input matrix. shape: (# rows, # cols)
    @return: 2D non sparse array. cosin similarity of each row in x. shape: (# rows, # rows)
    '''
    sim = x.dot(x.T)
    sim = sim.toarray() + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def pearson_sim(x, epsilon=1e-9):
    '''
    @param x: 2D (sparse) array. Input matrix. shape: (# rows, # cols)
    @return: 2D non sparse array. correlation coefficient of each row in x. shape: (# rows, # rows)
    '''
    num_cols = float(x.shape[1])
    meanx = x.sum(axis=1)/num_cols
    covx = np.array((x.dot(x.T))/num_cols - meanx.dot(meanx.T))

    stdx = np.sqrt(np.diag(covx))  # std of each item, shape: (# rows,)
    stdx = stdx[:, np.newaxis]  # shape: (# rows, 1)
    res = covx/(stdx.dot(stdx.T))
    # # Note that when one item of the rating matrix is all the same (usually it should be all 0 due to the train/test split process), the std of that column will be 0 and res will be nan for that column.
    res[np.isnan(res)] = 0

    return res


def shrunk_sim(x, sim_mat, lambda2=100):
    assert sim_mat.shape[0] == sim_mat.shape[1]

    sim_mat = sim_mat.copy()
    num_row = sim_mat.shape[0]
    on_cols = {}
    for i in range(0, num_row):
        on_cols[i] = set(x.getcol(i).indices)

    for i in range(0, num_row):
        for j in range(i+1, num_row):
            nij = len(on_cols[i].intersection(on_cols[j]))
            sim_mat[i][j] *= (float(nij)/(nij+lambda2))
            sim_mat[j][i] *= (float(nij)/(nij+lambda2))

    return sim_mat
