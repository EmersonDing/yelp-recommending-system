#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to pre-calculate k-nearest neighbor of item in the rating matrix
"""

import argparse
import sys

import numpy as np
from scipy import io


def parse_arg(argv):
    parser = argparse.ArgumentParser(description='Calculate k-neares neighbor of each item')
    parser.add_argument('-i', '--inpf', default='../processed_data/Stars_30000.mtx', help='input sparse matrix file of the rating matrix')
    parser.add_argument('-o', '--oupf', default='../processed_data/Stars_30000_neighbors.npy', help='input sparse matrix file of the rating matrix')
    return parser.parse_args(argv[1:])


def items_pearson_similarity(x):
    '''
    @param x: 2D sparse array. Input matrix. shape: (# users, # items).
    @return 2D non sparse array. correlation coefficient matrix of x. shape: (# items, # items)
    '''
    covx = column_covariance(x)
    stdx = np.sqrt(np.diag(covx))  # std of each item, shape: (# item,)
    stdx = stdx[np.newaxis,:]  # shape: (1, # of item)
    return covx/(stdx.T * stdx)


def column_covariance(x):
    '''
    @param x: 2D (sparse) array. Input matrix. shape: (# rows, # columns)
    @return: 2D non sparse array. Covariance matrix of x's columns. shape: (# columns, # columns)
    '''
    num_rows = float(x.shape[0])
    meanx = x.sum(axis=0)/num_rows
    res = np.array((x.T*x)/num_rows - meanx.T*meanx)
    return res


def shrunk_sim(x, sim_mat):
    assert sim_mat.shape[0] == sim_mat.shape[1]

    lambda2 = 100
    num_item = sim_mat.shape[0]
    rated_users = {}
    for i in range(0, num_item):
        rated_users[i] = set(x.getcol(i).indices)

    for i in range(i+1, num_item):
        for j in range(i+1, num_item):
            nij = rated_users[i].intersection(rated_users[j])
            sim_mat[i][j] *= (float(nij)/(nij+lambda2))

    return sim_mat


def knn(rating, k=50):
    sim_mat = items_pearson_similarity(rating)
    shrunk_sim_mat = shrunk_sim(rating, sim_mat)

    neighbors = np.zeros((sim_mat.shape[0], k), dtype=int)
    for i, row in enumerate(shrunk_sim_mat):
        neighbors[i] = np.argsort(row)[-1::-1][1:k+1]  # do not consider the item itself

    return neighbors

if __name__ == '__main__':
    args = parse_arg(sys.argv)
    rating = io.mmread(args.inpf).tocsc()

    neighbors = knn(rating, k=50)

    print("Saving to {}...".format(args.oupf))
    np.save(args.oupf, neighbors)
