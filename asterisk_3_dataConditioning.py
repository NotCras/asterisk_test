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
import numpy as np
import asterisk_0_prompts as prompts


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

    print("LOADING HAND DIMENSIONS")
    with open(".hand_dimensions") as csv_file:
        csv_reader_hands = csv.reader(csv_file, delimiter=',')
        #populating dictionaries with dimensions
        for row in csv_reader_hands:
            hand_name = row[0]
            hand_span = float(row[1])
            hand_depth = float(row[2])

            hand_spans[hand_name] = hand_span
            hand_depths[hand_name] = hand_depth

    #import csv file of data
    #for now, just write it out each time
    folder_path = "csv/"
    subject_name = "josh"
    hand = "2v2"
    #t = "none"
    #d = "a"
    #num = "3"

    #file_name = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num + ".csv"

    #total_path = folder_path + file_name

    if hand == "basic" or hand == "m2stiff" or hand == "vf":
        types = ["none"]
    else:
        types = prompts.type_options

    for t in types:
        if t == "none":
            directions = prompts.dir_options
        else:
            directions = prompts.dir_options_no_rot

        for d in directions:
            for num in ["1", "2", "3"]: #TODO: HANDLE MISSING FILES SOMEHOW prompts.trial_options:
                file_name = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num + ".csv"

                total_path = folder_path + file_name
                
                print(" ")
                print("READ IN DATA CSV")
                df = pd.read_csv(total_path, names=["roll", "pitch", "yaw", "x", "y", "z", "tmag", "rmag"])

                print(" ")
                print("CONVERT M TO MM")
                #convert m to mm in translational data
                df = df * [1, 1, 1, 1000, 1000, 1000, 1000, 1]
                df = df.round(4)

                print(" ")
                print("NORMALIZE DATA BY HAND DIMENSION")
                print(f"Hand span: {hand_spans[hand]}")
                print(f"Hand depth: {hand_depths[hand]}")
                #normalize translational data by hand span
                df = df / [1, 1, 1, #orientation data
                    hand_spans[hand] * 0.5, #x
                    hand_depths[hand], #y
                    1, #z - doesn't matter
                    1, #translational magnitude - don't use
                    1]
                df = df.round(4)

                print(" ")
                print("FILTER DATA WITH MOVING AVERAGE")
                #filter the data, by column
                rolling_window_size = 15
                df["f_x"] = df["x"].rolling(window=rolling_window_size).mean()
                df["f_y"] = df["y"].rolling(window=rolling_window_size).mean()
                df["f_rot_mag"] = df["rmag"].rolling(window=rolling_window_size).mean() #TODO: FILL IN THE NaN spots with values from unfiltered section
                df = df.round(4)

                #store in a new file
                print("GENERATING FILE FOR: " + file_name)
                new_file_name = "filtered/filt_" + file_name 
                df.to_csv(new_file_name, index=True, columns = ["f_x", "f_y", "f_rot_mag"])
                print("COMPLETED!")

