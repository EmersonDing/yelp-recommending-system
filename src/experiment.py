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
import matplotlib.pyplot as plt
import similarity
from model import Simple_sim
from model import Bias
from model import Neighbor
from model import Factor
from model import Integrated
from scipy import sparse


def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Prepare rating matrix')
    parser.add_argument('-i', '--inpf', default='../processed_data/Stars_Latest.pkl', help='pickle file containing train/test rating matrix')
    parser.add_argument('-ri', '--raw_inpf', default='../raw_data/Stars_Latest.csv', help='csv file for rating matrix, used to genrate inpf')
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
    result = []
    if user_based:
        model = modelCls(**model_args)
        model.train(train_mat, test_mat)
        train_prediction = model.predict(train_mat, train_mat)
        test_prediction = model.predict(train_mat, test_mat)
        result.append(get_mse(train_prediction, train_mat))
        result.append(get_mse(test_prediction, test_mat))

    if item_based:
        train_matT, test_matT = train_mat.T.tocsr(), test_mat.T.tocsr()
        model = modelCls(**model_args)
        model.train(train_matT, test_matT)
        train_prediction = model.predict(train_matT, train_matT)
        test_prediction = model.predict(train_matT, test_matT)
        result.append(get_mse(train_prediction, train_mat))
        result.append(get_mse(test_prediction, test_mat))

    return result

def diffIteration():
    iterations = [5, 10, 15, 25, 50, 100]
    user_based_Bias_Training = []
    user_based_Bias_Testing = []
    item_based_Bias_Training = []
    item_based_Bias_Testing = []
    user_based_Neighbor_Training = []
    user_based_Neighbor_Testing = []
    item_based_Neighbor_Training = []
    item_based_Neighbor_Testing = []
    user_based_Factor_Training = []
    user_based_Factor_Testing = []
    user_based_Integrated_Training = []
    user_based_Integrated_Testing = []
    item_based_Integrated_Training = []
    item_based_Integrated_Testing = []
    for iteration in iterations:
        result = doIt(Bias, iteration=iteration)
        user_based_Bias_Training.append(result[0])
        user_based_Bias_Testing.append(result[1])
        item_based_Bias_Training.append(result[2])
        item_based_Bias_Testing.append(result[3])
        result = doIt(Neighbor, sim_fn=similarity.cosine_sim, k=100, iteration=iteration)
        user_based_Neighbor_Training.append(result[0])
        user_based_Neighbor_Testing.append(result[1])
        item_based_Neighbor_Training.append(result[2])
        item_based_Neighbor_Testing.append(result[3])
        result = doIt(Factor, item_based=False, emb_dim=100, iteration=iteration)
        user_based_Factor_Training.append(result[0])
        user_based_Factor_Testing.append(result[1])
        result = doIt(Integrated, with_c=True, sim_fn=similarity.cosine_sim, k=100, with_y=True, emb_dim=100, iteration=5)
        user_based_Integrated_Training.append(result[0])
        user_based_Integrated_Testing.append(result[1])
        item_based_Integrated_Training.append(result[2])
        item_based_Integrated_Testing.append(result[3])

    # print result
    print("Iteration number: " , result)
    print("user_based_Bias_Training: " , user_based_Bias_Training)
    print("user_based_Bias_Testing: " , user_based_Bias_Testing)
    print("item_based_Bias_Training: " , item_based_Bias_Training)
    print("item_based_Bias_Testing: " , item_based_Bias_Testing)
    print("user_based_Neighbor_Training: " , user_based_Neighbor_Training)
    print("user_based_Neighbor_Testing: " , user_based_Neighbor_Testing)
    print("item_based_Neighbor_Training: " , item_based_Neighbor_Training)
    print("item_based_Neighbor_Testing: " , item_based_Neighbor_Testing)
    print("user_based_Factor_Training: " , user_based_Factor_Training)
    print("user_based_Factor_Testing: " , user_based_Factor_Testing)
    print("user_based_Integrated_Training: " , user_based_Integrated_Training)
    print("user_based_Integrated_Testing: " , user_based_Integrated_Testing)
    print("item_based_Integrated_Training: " , item_based_Integrated_Training)
    print("item_based_Integrated_Testing: " , item_based_Integrated_Testing)

    # printPlotDiffIteration(iterations, user_based_Bias_Training, item_based_Bias_Training, user_based_Neighbor_Training,
    #                    item_based_Neighbor_Training, user_based_Factor_Training)
    # printPlotTrainTest(iterations, user_based_Bias_Training, user_based_Bias_Testing, user_based_Neighbor_Training, user_based_Neighbor_Testing)
    printPlotIntegrated(iterations, user_based_Neighbor_Training, user_based_Bias_Training, user_based_Integrated_Training)

def diffKValue():
    result = []
    user_based_Neighbor = []
    item_based_Neighbor = []
    kSet = [1, 5, 10, 25, 50, 100]
    for k in kSet:
        result = doIt(Neighbor, sim_fn=similarity.cosine_sim, k=k, iteration=10)
        user_based_Neighbor.append(result[0])
        item_based_Neighbor.append(result[2])
    printPlotDiffK(kSet, user_based_Neighbor, item_based_Neighbor)

def printPlotDiffIteration(iterations, x1, y1, x2, y2, x3):
    plt.plot(iterations, x1, color = 'red', label = 'user_based_bias')
    plt.plot(iterations, y1, color = 'orange', label = 'item_based_bias')
    plt.plot(iterations, x2, color = 'pink', label = 'user_based_neighbor')
    plt.plot(iterations, y2, color = 'blue', label = 'item_based_neighbor')
    plt.plot(iterations, x3, color = 'green', label = 'factor')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Square Error')
    plt.legend(loc='upper left')
    plt.show()

def printPlotTrainTest(iterations, x1, y1, x2, y2):
    plt.plot(iterations, x1, color = 'red', label = 'bias_training')
    plt.plot(iterations, y1, color = 'orange', label = 'bias_testing')
    plt.plot(iterations, x2, color = 'pink', label = 'neighbor_training')
    plt.plot(iterations, y2, color = 'blue', label = 'neighbor_testing')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Square Error')
    plt.legend(loc='upper left')
    plt.show()

def printPlotIntegrated(iterations, x, y, z):
    plt.plot(iterations, x, color='red', label='bias')
    plt.plot(iterations, y, color='orange', label='neighbor')
    plt.plot(iterations, z, color='pink', label='integrated')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Square Error')
    plt.legend(loc='upper left')
    plt.show()

def printPlotDiffK(k, x, y):
    line1 = plt.plot(k, x, color = 'red', label = 'user_based_neighbor')
    line2 = plt.plot(k, y, color = 'blue', label = 'item_based_neighbor')
    plt.xlabel('k')
    plt.ylabel('Mean Square Error')
    plt.legend(loc='upper left')
    plt.show()

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
        sparsity = float(train_mat.getnnz() + test_mat.getnnz()) / (train_mat.shape[0] * train_mat.shape[1])
        print('testing {}'.format(args.inpf))
        print('Sparsity: {}'.format(sparsity))

    # printPlot([1,2,3], [4,6,9], [2,3,4], [7,8,9], [1,2,3], [4,3,2])
    diffIteration()
    # diffKValue()

