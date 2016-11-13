# -*- coding: utf-8 -*-
"""
An implementation of matrix factorization
Source:
http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#the-mathematics-of-matrix-factorization
"""

try:
    import numpy as np
except:
    print 'This implementation requires the numpy module.'
    exit(0)


def matrix_factorization(r, p, q, K, alpha=0.0002, beta=0.02, steps=5000, tol=0.001):
    """
    An implementation of matrix factorization
    :param r:       a matrix to be factorized, dimension N x M
    :param p:       an initial matrix of dimension N x K
    :param q:       an initial matrix of dimension M x K
    :param K:       the number of latent features
    :param alpha:   the learning rate
    :param beta:    the regularization parameter
    :param steps:   the maximum number of steps to perform the optimisation
    :param tol:     the maximum tolerance of error.
    :return:        the final matrices P and Q
    """
    q = q.T
    for step in xrange(steps):
        for i in xrange(len(r)):
            for j in xrange(len(r[i])):
                if r[i][j] > 0:
                    eij = r[i][j] - np.dot(p[i, :], q[:, j])
                    for k in xrange(K):
                        p[i][k] = p[i][k] + alpha * (2 * eij * q[k][j] - beta * p[i][k])
                        q[k][j] = q[k][j] + alpha * (2 * eij * p[i][k] - beta * q[k][j])
        eR = np.dot(p, q)
        e = 0
        for i in xrange(len(r)):
            for j in xrange(len(r[i])):
                if r[i][j] > 0:
                    e = e + pow(r[i][j] - np.dot(p[i, :], q[:, j]), 2)
                    for k in xrange(K):
                        e = e + (beta / 2) * (pow(p[i][k], 2) + pow(q[k][j], 2))
        if e < tol:
            break
    return p, q.T


if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = np.dot(nP, nQ.T)
    print nR
    #print nP
    #print nQ
