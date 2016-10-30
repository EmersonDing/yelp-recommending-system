#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare user-item rating matrix
"""

import pandas as pd
import argparse
import sys
from scipy import sparse, io


def parse_arg(argv):
    '''
    parsing cli arguments
    '''
    parser = argparse.ArgumentParser(description='Prepare rating matrix')
    parser.add_argument('-i', '--inpf', default='../raw_data/Stars_30000.csv', help='csv input file for start matrix')
    parser.add_argument('-o', '--oupf', default='../processed_data/Stars_30000.mtx', help='scipy sparse matrix file containing star matrix')
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_arg(sys.argv)
    df = pd.read_csv(args.inpf)
    print(df.head())

    # extract and user id
    # TODO: Some user id are stange, "(Also" is a use id?
    # print(df['user_id'])

    user_id = {uid: i for i, uid in enumerate(set(df["user_id"]))}
    num_user = len(user_id)
    print('Total unique user id: {}'.format(num_user))

    # extract business id
    business_id = {bid: i for i, bid in enumerate(set(df["business_id"]))}
    num_business = len(business_id)
    print('Total unique business id: {}'.format(num_business))

    # construct scipy sparse csr_matrix
    # TODO: csr is slow for row indexing, might need to change to other sparse format later
    data = []
    row_ind = []
    col_ind = []
    for _, record in df.iterrows():
        uid, bid, rating = user_id[record[0]], business_id[record[1]], record[2]
        row_ind.append(uid)
        col_ind.append(bid)
        data.append(rating)

    us_bs_matrix = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(num_user, num_business), dtype=int)

    print 'Average rating: ',us_bs_matrix[us_bs_matrix > 0].mean()

    # output to mtx file
    print 'Saving...'
    io.mmwrite(args.oupf, us_bs_matrix)
