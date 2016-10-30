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
    return parser.parse_args(argv[1:])


def items_pearson_similarity(x):
    '''
    @param x: 2D sparse array. Input matrix. Rows stand for users, and columns stands for items.
    @return ndarray. correlation coefficient matrix of x.
    '''
    covx = column_covariance(x)
    stdx = np.sqrt(np.diag(covx))  # variance of each column, shape: (column # of x,)
    stdx = stdx[np.newaxis,:]  # shape: (1, column # of x)
    return covx/(stdx.T * stdx)


def column_covariance(x):
    '''
    @param x: 2D (sparse) array. Input matrix.
    @return: 2D (sparse) array. Covariance matrix of x's columns.
    '''
    x = x.T
    mean = x.sum(axis=1)/float(x.shape[1])
    x = x - mean
    return x * x.T

if __name__ == '__main__':
    # TODO: to be implemented
    args = parse_arg(sys.argv)
    rating = io.mmread(args.inpf).tocsr()
    sim_mat = items_pearson_similarity(rating)
    print(sim_mat.shape)
