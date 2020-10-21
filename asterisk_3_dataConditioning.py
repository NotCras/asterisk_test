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
import numpy as n

def convert_to_mm(val):
    return val * 1000 #TODO: make it only return a certain number of sig figs

def normalize_vertical_data(val, depth):
    return val / depth

def normalize_horizontal_data(val, span):
    return val / span

def normalize_diagonal_data(xval, yval, span, depth):
    x_norm = normalize_horizontal_data(xval, span)
    y_norm = normalize_vertical_data(yval, depth)
    return x_norm, y_norm

def filter_data(data, window_length = 3):
    #moving average as seen here: https://www.kite.com/python/answers/how-to-find-moving-average-from-data-points-in-python
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
    return moving_averages #test this out - double check it


if __name__ == "__main__":
    pass
