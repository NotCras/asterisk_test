"""
This file will...
0) look at the csv files in the csv/ folder
1) convert to mm (starts in m)
2) normalize the translation data by hand span
3) run a moving average filter over the data
4) save edited csv file in the conditioned/ folder

"""

import csv

def normalize_data(val, hand_span):
    return val / hand_span

def filter_data(data):
    #moving average
    filtered_data = data

    return filtered_data

def remove_clusters(data):
    #remove clusters
    declustered_data = data

    return declustered_data

def resample_data(data):
    #resample data
    resampled_data = data

    return resampled_data


if __name__ == "__main__":

pass
