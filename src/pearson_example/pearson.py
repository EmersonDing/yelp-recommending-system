'''
Source:
http://stackoverflow.com/questions/3949226/calculating-pearson-correlation-and-significance-in-python
'''

import numpy as np
import math


def get_pearson_correlation(first_feature_vector=[], second_feature_vector=[], length_of_featureset=0):
    '''
    An implementation for pearson correlation based on sparse vector. The vectors here are expressed as
    a list of tuples expressed as (index, value). The two sparse vectors can be of different length but
    over all vector size will have to be same.
    :param first_feature_vector: array of tuple
    :param second_feature_vector: array of tuple
    :param length_of_featureset: int
    :return: float
    '''
    indexed_feature_dict = {}
    if first_feature_vector == [] or second_feature_vector == [] or length_of_featureset == 0:
        raise ValueError("Empty feature vectors or zero length of featureset in get_pearson_correlation")

    sum_a = sum(value for index, value in first_feature_vector)
    sum_b = sum(value for index, value in second_feature_vector)

    avg_a = float(sum_a) / length_of_featureset
    avg_b = float(sum_b) / length_of_featureset

    mean_sq_error_a = np.sqrt((sum((value - avg_a) ** 2 for index, value in first_feature_vector)) + ((
        length_of_featureset - len(first_feature_vector)) * ((0 - avg_a) ** 2)))
    mean_sq_error_b = np.sqrt((sum((value - avg_b) ** 2 for index, value in second_feature_vector)) + ((
        length_of_featureset - len(second_feature_vector)) * ((0 - avg_b) ** 2)))

    covariance_a_b = 0

    #calculate covariance for the sparse vectors
    for tuple in first_feature_vector:
        if len(tuple) != 2:
            raise ValueError("Invalid feature frequency tuple in featureVector: %s") % (tuple,)
        indexed_feature_dict[tuple[0]] = tuple[1]
    count_of_features = 0
    for tuple in second_feature_vector:
        count_of_features += 1
        if len(tuple) != 2:
            raise ValueError("Invalid feature frequency tuple in featureVector: %s") % (tuple,)
        if tuple[0] in indexed_feature_dict:
            covariance_a_b += ((indexed_feature_dict[tuple[0]] - avg_a) * (tuple[1] - avg_b))
            del (indexed_feature_dict[tuple[0]])
        else:
            covariance_a_b += (0 - avg_a) * (tuple[1] - avg_b)

    for index in indexed_feature_dict:
        count_of_features += 1
        covariance_a_b += (indexed_feature_dict[index] - avg_a) * (0 - avg_b)

    #adjust covariance with rest of vector with 0 value
    covariance_a_b += (length_of_featureset - count_of_features) * -avg_a * -avg_b

    if mean_sq_error_a == 0 or mean_sq_error_b == 0:
        return -1
    else:
        return float(covariance_a_b) / (mean_sq_error_a * mean_sq_error_b)


def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    '''
    Compute Pearson Correlation Coefficient.
    (only work on single array)
    :param x: numpy array
    :param y: numpy array
    :return: int
    '''
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


if __name__=='__main__':
    X1 = np.array([1, 2, 3])
    Y1 = np.array([1, 5, 7])

    print np.corrcoef(X1, Y1)
    print pearson_def(X1, Y1)

    vector_a = [(1, 1), (2, 2), (3, 3)]
    vector_b = [(1, 1), (2, 5), (3, 7)]
    print get_pearson_correlation(vector_a, vector_b, 3)
