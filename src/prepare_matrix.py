#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare user-item rating matrix
"""

import pandas as pd
import argparse
import sys
from scipy import sparse
import numpy as np
import pickle


def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Prepare rating matrix')
    parser.add_argument('-i', '--inpf', default='../raw_data/Stars_Top_Users.csv', help='csv input file for start matrix')
    parser.add_argument('-o', '--oupf', default='../processed_data/Stars_Top_Users.mtx', help='scipy sparse matrix file containing star matrix')
    parser.add_argument('-s', '--split', type=float, default=0.1, help='ratio of testing split data, e.g. 0.1 will split 1/10 (usr, business) ratings as testing pairs.')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='random seed to split tainging and testing data')
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_arg(sys.argv)
    np.random.seed(args.random_seed)
    df = pd.read_csv(args.inpf)

    # extract and user id
    # TODO: Some user id are stange, "(Also" is a use id?

    user_id = {uid: i for i, uid in enumerate(set(df["user_id"]))}
    num_user = len(user_id)
    print('Total unique user id: {}'.format(num_user))

    # extract business id
    business_id = {bid: i for i, bid in enumerate(set(df["business_id"]))}
    num_business = len(business_id)
    print('Total unique business id: {}'.format(num_business))

    data = []
    count= []
    row_ind = []
    col_ind = []
    for _, record in df.iterrows():
        uid, bid, rating = user_id[record[0]], business_id[record[1]], record[2]
        row_ind.append(uid)
        col_ind.append(bid)
        data.append(rating)
        count.append(1)

    # One user might rate one business multiple times, we calculate the avrage for that.
    count_matrix = sparse.csr_matrix((count, (row_ind, col_ind)), shape=(num_user, num_business), dtype=float)
    us_bs_matrix = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(num_user, num_business), dtype=float)

    # Do the average
    count_matrix.data = 1/count_matrix.data
    us_bs_matrix = us_bs_matrix.multiply(count_matrix)

    # Split train, test data by randomly picking "split" part of the (user, rating) pairs as testing pairs
    ub_coo = us_bs_matrix.tocoo()
    data_row_col = np.random.permutation([[r,c,d] for r,c,d in zip(ub_coo.data, ub_coo.row, ub_coo.col)])

    split = int(args.split * len(data_row_col))
    test = data_row_col[:split]
    train = data_row_col[split:]

    train_mat = sparse.coo_matrix((train[:,0], (train[:,1], train[:,2])), shape=(num_user, num_business), dtype=float)
    test_mat = sparse.coo_matrix((test[:,0], (test[:,1], test[:,2])), shape=(num_user, num_business), dtype=float)

    print("Training pairs: {}".format(train_mat.getnnz()))
    print("Testing pairs: {}".format(test_mat.getnnz()))
    # output to mtx file
    with open(args.oupf, 'w') as f:
        pickle.dump({'train':train_mat, 'test': test_mat}, f)
