#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to test different algorithms
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import similarity
from model import Simple_sim
from model import Bias
from model import Neighbor
from model import Factor
from scipy import sparse


def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Prepare rating matrix')
    parser.add_argument('-i', '--inpf', default='../processed_data/Stars_Top_Users.pkl', help='pickle file containing train/test rating matrix')
    parser.add_argument('-ri', '--raw_inpf', default='../raw_data/Stars_Top_Users.csv', help='csv file for rating matrix, used to genrate inpf')
    parser.add_argument('-s', '--split', type=float, default=0.1, help='ratio of testing split data, e.g. 0.1 will split 1/10 (usr, item) ratings as testing pairs.')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='random seed to split tainging and testing data')
    return parser.parse_args(argv[1:])


def read_data(df):
    user_id = {uid: i for i, uid in enumerate(set(df["user_id"]))}
    item_id = {iid: i for i, iid in enumerate(set(df["business_id"]))}

    n_users = len(user_id)
    n_items = len(item_id)

    ratings = []
    count= []
    row_ind = []
    col_ind = []
    for _, row in df.iterrows():
        row_ind.append(user_id[row[0]])
        col_ind.append(item_id[row[1]])
        ratings.append(row[2])
        count.append(1)

    # One user might rate one item multiple times, we calculate the avrage for that.
    count_mat = sparse.csr_matrix((count, (row_ind, col_ind)), shape=(n_users, n_items), dtype=float)
    ui_mat = sparse.csr_matrix((ratings, (row_ind, col_ind)), shape=(n_users, n_items), dtype=float)

    # Do the average
    count_mat.data = 1/count_mat.data
    ui_mat = ui_mat.multiply(count_mat)

    return ui_mat


def train_test_split(ui_mat, split=0.1):
    train_mat = ui_mat.copy()
    test_mat = ui_mat.copy()
    for i in range(ui_mat.shape[0]):
        cols = ui_mat.getrow(i).indices
        cols = np.random.permutation(cols)
        cut = int(split*len(cols))
        test_ind = cols[:cut]
        train_ind = cols[cut:]

        test_mat[i, train_ind] = 0
        train_mat[i, test_ind] = 0

    train_mat.eliminate_zeros()
    test_mat.eliminate_zeros()

    # Training and testing should be disjoint
    assert((train_mat.multiply(test_mat)).sum() == 0)
    return train_mat, test_mat


def preprocess_data(fname):
    # TODO: Some user id are stange, "(Also" is a use id?
    df = pd.read_csv(fname)
    # print df.head()

    user_id = {uid: i for i, uid in enumerate(set(df["user_id"]))}
    item_id = {iid: i for i, iid in enumerate(set(df["business_id"]))}

    n_users = len(user_id)
    n_items = len(item_id)
    ratings = []
    count= []
    row_ind = []
    col_ind = []
    for _, row in df.iterrows():
        row_ind.append(user_id[row[0]])
        col_ind.append(item_id[row[1]])
        ratings.append(row[2])
        count.append(1)

    # One user might rate one item multiple times, we calculate the avrage for that.
    count_mat = sparse.csr_matrix((count, (row_ind, col_ind)), shape=(n_users, n_items), dtype=float)
    ui_mat = sparse.csr_matrix((ratings, (row_ind, col_ind)), shape=(n_users, n_items), dtype=float)

    # Do the average
    count_mat.data = 1/count_mat.data
    ui_mat = ui_mat.multiply(count_mat)

    return ui_mat


def get_mse(pred, actual):
    '''Calculate mean square error.
    @param pred: 1D array. Length of pred should be equal to actual.getnnz()
    @actual: 2D sparse matrix.
    '''
    pred = np.asarray(pred).flatten()
    actual = np.asarray(actual[actual.nonzero()]).flatten()
    assert(len(pred)==len(actual))
    return np.sum((pred-actual)**2)/len(pred)


def doIt(modelCls, user_based=True, item_based=True, **model_args):
    '''Compare different models.
    @modelCls class: The class of the target model.
    @param doItemBased: boolean. WHether to do item based method by transposing the matrix.
    @param model_args: args for initializing modelCls.
    '''
    print
    print('='*20)
    print('Model: {}'.format(modelCls.__name__))
    print('Args: {}'.format(model_args))
    if user_based:
        user_model = modelCls(**model_args)
        user_model.train(train_mat, test_mat)
        user_prediction = user_model.predict(train_mat, test_mat)
        print('User-based CF MSE: {}'.format(get_mse(user_prediction, test_mat)))

    if item_based:
        train_matT, test_matT = train_mat.T.tocsr(), test_mat.T.tocsr()
        item_model = modelCls(**model_args)
        item_model.train(train_matT, test_matT)
        item_prediction = item_model.predict(train_matT, test_matT)
        print('Item-based CF MSE: {}'.format(get_mse(item_prediction, test_matT)))

if __name__ == '__main__':

    args = parse_arg(sys.argv)
    np.random.seed(args.random_seed)

    if not os.path.isfile(args.inpf):
        ui_mat = preprocess_data(args.raw_inpf)
        train_mat, test_mat = train_test_split(ui_mat, 0.1)
        with open(args.inpf, 'wb') as f:
            print("Saving train/test matrix ..")
            print("{} users".format(ui_mat.shape[0]))
            print("{} items".format(ui_mat.shape[1]))
            print("{} training pairs".format(train_mat.getnnz()))
            print("{} testing pairs".format(test_mat.getnnz()))

            pickle.dump({'train':train_mat, 'test': test_mat}, f)
    else:
        with open(args.inpf, 'rb') as f:
            data = pickle.load(f)
            train_mat, test_mat = data['train'], data['test']

    doIt(Simple_sim, sim_fn=similarity.cosine_sim)
    doIt(Bias, iteration=10)
    doIt(Neighbor, sim_fn=similarity.cosine_sim, k=100, iteration=10)
    doIt(Factor, item_based=False, emb_dim=100, iteration=10)
