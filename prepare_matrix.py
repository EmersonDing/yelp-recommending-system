# -*- coding: utf-8 -*-
"""
@author: Chien-Pin Chen

A script to prepare rating matrix

"""

import pandas as pd
import numpy as np
import os

'''
Load file
Assume you put csv file with this script in the same directory
'''
dir_path = os.getcwd()
tot_path = os.path.join(dir_path, 'Stars_30000.csv')
# note: Stars_3000.csv not in repo)
#print tot_path
df = pd.read_csv(tot_path)
print df.head()


'''
extract business id and user id
'''
df1 = df["user_id"]
us_lst = list(set(df1))
us_lst.sort()
print 'Total unique user id: ', len(us_lst)
#print us_lst.index('--65q1FpAL_UQtVZ2PTGew')


df2 = df["business_id"]
bs_lst = list(set(df2))
bs_lst.sort()
print 'Total unique business id: ', len(bs_lst)
#print bs_lst.index('VZYMInkjRJVHwXVFqeoMWg')


'''
construct numpy matrix
'''
us_bs_matrix = np.zeros((len(us_lst), len(bs_lst)), dtype=int)
print 'The shape of matrix: ', us_bs_matrix.shape

for index, row in df.iterrows():
    r_ind = us_lst.index(row[0])
    c_ind = bs_lst.index(row[1])
    us_bs_matrix[r_ind][c_ind] = row[2]

print 'Average rating: ',us_bs_matrix[us_bs_matrix > 0].mean()

'''
output matrix to csv file (~ 200 MB)
'''
print 'Saving... take about 1~2 minutes'
np.savetxt('rating_matrix.csv', us_bs_matrix, fmt='%d', delimiter=",")
print 'Done!'

