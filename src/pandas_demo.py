import pandas as pd
"""
This file demostrate how to read file from
"""

pd.set_option('display.mpl_style', 'default')


class Star:
    def __init__(self):
        df = pd.read_csv

    ## set path saved with Stars.csv
    def readCSV(self, path):
        self.df = pd.read_csv(path)

if __name__ == '__main__':
    ## initial star object
    ob = Star()

    ## initial csv file
    ob.readCSV("/Users/emerson/Documents/Github/yelp_recommending_system/Stars.csv")

    ## print 20th row to 40th
    # print(ob.df[20:40])

    ## print 20th row to 40th of user_id column
    # print(ob.df['user_id'][20:40])

    ## print all of the reviews with 4 star
    print(ob.df[ob.df['stars'].isin([4])])
