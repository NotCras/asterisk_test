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
import asterisk_0_prompts as prompts

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

def filter_data(data_column, window_length = 3):
    #moving average as seen here: https://www.kite.com/python/answers/how-to-find-moving-average-from-data-points-in-python
    cumulative_sum = np.cumsum(np.insert(data_column, 0, 0))
    moving_averages = (cumulative_sum[window_length:] - cumulative_sum[:-window_length]) / window_length
    return moving_averages #test this out - double check it


if __name__ == "__main__":
    #import hand span data
    hand_spans = dict()
    hand_depths = dict()

    with open(".hand_dimensions") as csv_file:
        csv_reader_hands = csv.reader(csv_file, delimiter=',')
        #populating dictionaries with dimensions
        for row in csv_reader_hands:
            hand_name = row[0]
            hand_span = row[1]
            hand_depth = row[2]

            hand_spans[hand_name] = hand_span
            hand_depths[hand_name] = hand_depth

    #import csv file of data
    #for now, just write it out each time
    folder_path = "csv/"
    subject_name = "josh"
    hand = "2v2"
    t = "none"
    d = "a"
    num = "3"

    file_name = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num + ".csv"

    total_path = folder_path + file_name

    with open(total_path) as data_file:
        csv_reader_data = csv.reader(data_file, deliminater=',')

        for 


        #go through each column - mm and normalize


        #filter the data, by column


        #store in a new file


    pass
