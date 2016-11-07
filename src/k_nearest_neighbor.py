#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to pre-calculate k-nearest neighbor of item in the rating matrix
Should be merge into model.py/similarity.py
"""

import argparse
import sys

import numpy as np
import pickle


def parse_arg(argv):
    parser = argparse.ArgumentParser(description='Calculate k-neares neighbor of each item')
    parser.add_argument('-i', '--inpf', default='../processed_data/Stars_30000.pkl', help='input pickle file containint train/test sparse rating matrix')
    parser.add_argument('-o', '--oupf', default='../processed_data/Stars_30000.npy', help='')
    return parser.parse_args(argv[1:])


def items_pearson_similarity(rating):
    '''
    @param rating: 2D sparse array. Rating matrix. shape: (# users, # items).
    @return 2D non sparse array. Item similarity matrix. shape: (# items, # items)
    '''
    covx = column_covariance(rating)
    stdx = np.sqrt(np.diag(covx))  # std of each item, shape: (# item,)
    stdx = stdx[np.newaxis,:]  # shape: (1, # of item)
    res = covx/(stdx.T * stdx)
    # Note that when one item of the rating matrix is all the same (usually it should be all 0 due to the train/test split process), the std of that column will be 0 and res will be nan for that column.
    res[np.isnan(res)] = 0
    return res


def column_covariance(x):
    '''
    @param x: 2D (sparse) array. Input matrix. shape: (# rows, # columns)
    @return: 2D non sparse array. Covariance matrix of x's columns. shape: (# columns, # columns)
    '''
    num_rows = float(x.shape[0])
    meanx = x.sum(axis=0)/num_rows
    res = np.array((x.T*x)/num_rows - meanx.T*meanx)
    return res


def shrunk_sim(rating, sim_mat):
    '''
    @param rating: 2D (sparse) array. Rating matrix. shape: (# users, # items)
    @param sim_mat: 2D non sparse array. Item similarity matrix. shape: (# items, # items)
    @return: 2D non sparse array. Shrunk version of sim_mat. shape: (# items, # items)
    '''
    assert sim_mat.shape[0] == sim_mat.shape[1]

    lambda2 = 100
    num_item = sim_mat.shape[0]
    rated_users = {}
    for i in range(0, num_item):
        rated_users[i] = set(rating.getcol(i).indices)

    for i in range(i+1, num_item):
        for j in range(i+1, num_item):
            nij = rated_users[i].intersection(rated_users[j])
            sim_mat[i][j] *= (float(nij)/(nij+lambda2))

    return sim_mat


def find_item_neighbors(rating, k=50):
    '''
    @param rating: 2D (sparse) array. Rating matrix. shape: (# users, # items)
    @param sim_mat: 2D non sparse array. correlation coefficient of items. shape: (# items, # items)
    @return: 2D non sparse array. Shrunk version of sim_mat. shape: (# items, # items)
    '''
    sim_mat = items_pearson_similarity(rating)
    # TODO check if non-shrunk is worse
    shrunk_sim_mat = shrunk_sim(rating, sim_mat)

    # TODO check if we find neighbors with absolute value could be better. Cause from the view of information gain, most unsimilar item is as informative as most similar item. 

    neighbors = np.zeros((sim_mat.shape[0], k), dtype=int)
    for i, row in enumerate(shrunk_sim_mat):
        neighbors[i] = np.argsort(row)[-1::-1][1:k+1]  # do not consider the item itself

    return neighbors


def find_Ru(rating, neighbors):
    for (i, r_row) in enumerate(rating):
        print set(r_row.indices).intersection(set(neighbors[i]))
        sys.exit(-1)


if __name__ == '__main__':
    args = parse_arg(sys.argv)
    with open(args.inpf, 'r') as f:
        rating = pickle.load(f)['train']

    # find neighbor is column based, use csc format
    neighbors = find_item_neighbors(rating.tocsc(), k=500)

    # # find Ru is row based, use csr format
    # find_Ru(rating.tocsr(), neighbors)

    # print("Saving to {}...".format(args.oupf))
    # np.save(args.oupf, neighbors)
