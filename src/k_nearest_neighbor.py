#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to pre-calculate k-nearest neighbor of item in the rating matrix
"""

import sys
from scipy import io
import argparse


def parse_arg(argv):
    parser = argparse.ArgumentParser(description='Calculate k-neares neighbor of each item')
    parser.add_argument('-i', '--inpf', default='../processed_data/Stars_30000.mtx', help='input sparse matrix file of the rating matrix')
    return parser.parse_args(argv[1:])

if __name__ == '__main__':
    # TODO: to be implemented
    args = parse_arg(sys.argv)
    rating = io.mmread(args.inpf).tocsr()
    print(rating.getcol(0))
