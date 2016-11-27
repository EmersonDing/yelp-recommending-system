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
    parser.add_argument('-i', '--inpf', default='../processed_data/Stars_item_oriented.pkl', help='pickle file containing train/test rating matrix')
    parser.add_argument('-ri', '--raw_inpf', default='../raw_data/Stars_item_oriented.csv', help='csv file for rating matrix, used to genrate inpf')
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
    user_based_Neighbor_Training_no_x = []
    user_based_Neighbor_Testing_no_x = []
    item_based_Neighbor_Training_no_x = []
    item_based_Neighbor_Testing_no_x = []
    user_based_Factor_Training = []
    user_based_Factor_Testing = []
    user_based_Factor_Training_no_c = []
    user_based_Factor_Testing_no_c = []
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
        result = doIt(Neighbor, with_c=True, sim_fn=similarity.cosine_sim, k=50, iteration=iteration)
        user_based_Neighbor_Training.append(result[0])
        user_based_Neighbor_Testing.append(result[1])
        item_based_Neighbor_Training.append(result[2])
        item_based_Neighbor_Testing.append(result[3])
        result = doIt(Neighbor, with_c=False, sim_fn=similarity.cosine_sim, k=50, iteration=iteration)
        user_based_Neighbor_Training_no_x.append(result[0])
        user_based_Neighbor_Testing_no_x.append(result[1])
        item_based_Neighbor_Training_no_x.append(result[2])
        item_based_Neighbor_Testing_no_x.append(result[3])
        result = doIt(Factor, with_y=True, emb_dim=10, iteration=iteration)
        user_based_Factor_Training.append(result[0])
        user_based_Factor_Testing.append(result[1])
        result = doIt(Factor, with_y=False, emb_dim=10, iteration=iteration)
        user_based_Factor_Training_no_c.append(result[0])
        user_based_Factor_Testing_no_c.append(result[1])
        result = doIt(Integrated, with_c=True, sim_fn=similarity.cosine_sim, k=50, with_y=True, emb_dim=10, iteration=iteration)
        user_based_Integrated_Training.append(result[0])
        user_based_Integrated_Testing.append(result[1])
        item_based_Integrated_Training.append(result[2])
        item_based_Integrated_Testing.append(result[3])

    # print result
    print("Iteration number: " , iterations)
    print("user_based_Bias_Training: " , user_based_Bias_Training)
    print("user_based_Bias_Testing: " , user_based_Bias_Testing)
    print("item_based_Bias_Training: " , item_based_Bias_Training)
    print("item_based_Bias_Testing: " , item_based_Bias_Testing)
    print("user_based_Neighbor_Training_y: " , user_based_Neighbor_Training)
    print("user_based_Neighbor_Testing_y: " , user_based_Neighbor_Testing)
    print("item_based_Neighbor_Training_y: " , item_based_Neighbor_Training)
    print("item_based_Neighbor_Testing_y: " , item_based_Neighbor_Testing)
    print("user_based_Neighbor_Training_no_y: " , user_based_Neighbor_Training_no_x)
    print("user_based_Neighbor_Testing_no_y: " , user_based_Neighbor_Testing_no_x)
    print("item_based_Neighbor_Training_no_y: " , item_based_Neighbor_Training_no_x)
    print("item_based_Neighbor_Testing_no_y: " , item_based_Neighbor_Testing_no_x)
    print("user_based_Factor_Training_c: " , user_based_Factor_Training)
    print("user_based_Factor_Testing_c: " , user_based_Factor_Testing)
    print("user_based_Factor_Training_no_c: " , user_based_Factor_Training_no_c)
    print("user_based_Factor_Testing_no_c: " , user_based_Factor_Testing_no_c)
    print("user_based_Integrated_Training: " , user_based_Integrated_Training)
    print("user_based_Integrated_Testing: " , user_based_Integrated_Testing)
    print("item_based_Integrated_Training: " , item_based_Integrated_Training)
    print("item_based_Integrated_Testing: " , item_based_Integrated_Testing)

def diffKValue():
    result = []
    user_based_Neighbor = []
    user_based_Factor = []
    kSet = [1, 5, 10, 25, 50, 100]
    for k in kSet:
        result = doIt(Neighbor, with_c=True, sim_fn=similarity.cosine_sim, k=k, iteration=15)
        user_based_Neighbor.append(result[1])
    printPlotDiffK(kSet, user_based_Neighbor)

def diffEmbeddedDimension():
    Factor_y = []
    Factor_no_y = []
    kSet = [1, 5, 10, 25, 50, 100]
    for k in kSet:
        result = doIt(Factor, with_y=True, emb_dim=k, iteration=15)
        Factor_y.append(result[1])
        result = doIt(Factor, with_y=False, emb_dim=k, iteration=15)
        Factor_no_y.append(result[1])
    printPlotDiffEmbeddedDimension(kSet, Factor_y, Factor_no_y)

def printPlotDiffIteration(iterations, x1, x2, x3, x4):
    plt.plot(iterations, x1, color = 'red', label = 'bias')
    plt.plot(iterations, x2, color = 'blue', label = 'neighbor_with_y')
    plt.plot(iterations, x3, color = 'green', label = 'factor_without_c')
    plt.plot(iterations, x4, color = 'orange', label = 'integrated')
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

def printPlotDiffK(k, x):
    line1 = plt.plot(k, x, color = 'red', label = 'user_based_neighbor')
    plt.xlabel('k')
    plt.ylabel('Mean Square Error')
    plt.legend(loc='upper left')
    plt.show()

def printPlotDiffEmbeddedDimension(k, x, y):
    line1 = plt.plot(k, x, color = 'red', label = 'factor_with_y')
    line1 = plt.plot(k, y, color = 'blue', label = 'factor_without_y')
    plt.xlabel('embedded dimension')
    plt.ylabel('Mean Square Error')
    plt.legend(loc='upper left')
    plt.show()

def printPlotTwoDiff(k, x, y, label_x, label_y, coo_x):
    line1 = plt.plot(k, x, color = 'red', label = label_x)
    line1 = plt.plot(k, y, color = 'blue', label = label_y)
    plt.xlabel(coo_x)
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
    # diffEmbeddedDimension()
    # diffIteration()
    # diffKValue()
    # printPlotDiffIteration([5, 10, 15, 25, 50, 100], [0.90252341558002069, 0.87804352888419201, 0.87449905606789191, 0.87936163192542693, 0.89193660140760656, 0.89807255115300644], [0.89228608969168777, 0.86928905669794609, 0.86578550441649527, 0.86766916918794323, 0.8703647616501724, 0.8692765667665191], [0.90280740405052118, 0.8780799826409682, 0.87453107083178283, 0.87936130178641336, 0.89167800882026504, 0.89429544544024342])
    # printPlotTwoDiff([5, 10, 15, 25, 50, 100], [0.89155133903802553, 0.86848193585590938, 0.8643909388754214, 0.86452308214170925, 0.86353846395035161, 0.86016006496024955], [0.90099938533908175, 0.87593745004632717, 0.87080879062314009, 0.87230463042325168, 0.879699540241628, 0.88662072217269539], 'neighbor_with_y', 'neighbor_without_y', 'iteration time')
    # printPlotTwoDiff([5, 10, 15, 25, 50, 100],[0.90647392345648958, 0.87860970659724458, 0.87538288964632904, 0.87871370981797203, 0.88814789789925264, 0.91439048378141463],[0.90173184234791948, 0.87767627875669552, 0.87445678431240015, 0.8790071074845115,0.89273694359540434, 0.89143607988068874], 'factor_with_c', 'factor_without_c','iteration time')
    # printPlotDiffIteration([5, 10, 15, 25, 50, 100], [0.90252341558002069, 0.87804352888419201, 0.87449905606789191, 0.87936163192542693, 0.89193660140760656, 0.89807255115300644], [0.89155133903802553, 0.86848193585590938, 0.8643909388754214, 0.86452308214170925, 0.86353846395035161, 0.86016006496024955], [0.90173184234791948, 0.87767627875669552, 0.87445678431240015, 0.8790071074845115, 0.89273694359540434, 0.89143607988068874], [0.89622306324786227, 0.86878380117076481, 0.86240973069252025, 0.86290842278496283, 0.86281330057751959, 0.86084494420362268])
    # printPlotTwoDiff([5, 10, 15, 25, 50, 100], [0.61627274727278347, 0.54235431933133549, 0.49594567477639873, 0.42821490040172155, 0.31058038227338192, 0.15638576000142435], [0.89622306324786227, 0.86878380117076481, 0.86240973069252025, 0.86290842278496283, 0.86281330057751959, 0.86084494420362268], 'integrated_training', 'integrated_testing', 'iteration time')
    ## item oriented data plot
    # printPlotTwoDiff([5, 10, 15, 25, 50, 100], [0.59383826166764753, 0.62167932374313317, 0.64259715119277827, 0.68568211601086282, 0.71792286502001446, 0.77562758177478108], [0.62220693966890084, 0.60552281831444177, 0.63452754319943505, 0.64135634727002711, 0.6931455286933742, 0.70529617996513272], 'user_based_integrated_with_item', 'item_based_integrated_with_item', 'iteration time')
    ## user oritented data plot
    printPlotTwoDiff([5, 10, 15, 25, 50, 100], [0.74460274279392835, 0.72636920645000025, 0.71855240595565417, 0.71657615594565849, 0.72864280882519028, 0.74810324597548339], [0.89167520171005199, 0.94269429092118917, 0.97776664599321916, 1.0184491469219352, 1.0528746194378489, 1.0683040630516203], 'user_based_itegrated_with_user', 'item_based_itegrated_with_user', 'iteration time')