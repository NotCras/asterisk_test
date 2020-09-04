"""
This file will...
0) look at the csv files in the csv/ folder
1) convert to mm (starts in m)
2) normalize the translation data by hand span
3) run a moving average filter over the data
4) save edited csv file in the conditioned/ folder

"""

import csv
import pandas as pd

def convert_to_mm(val):
    return val * 1000 #TODO: make it only return a certain number of sig figs

def normalize_vertical_data(val, depth):
    return val / depth

def normalize_horizontal_data(val, span):
    return val / depth

def normalize_diagonal_data(val, span, depth):
    pass

def filter_data(data):
    #moving average
    filtered_data = data

    return filtered_data


if __name__ == "__main__":
    pass
