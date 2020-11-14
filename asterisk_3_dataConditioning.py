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
import asterisk_0_dataHelper as helper

def load_measurements():
    #import hand span data
    spans = dict()
    depths = dict()

    print("LOADING HAND MEASUREMENTS")
    with open(".hand_dimensions") as csv_file:
        csv_reader_hands = csv.reader(csv_file, delimiter=',')
        #populating dictionaries with dimensions
        for row in csv_reader_hands:
            hand_name = row[0]
            hand_span = float(row[1])
            hand_depth = float(row[2])

            spans[hand_name] = hand_span
            depths[hand_name] = hand_depth
    
    return spans, depths

def moving_average(df_to_filter, window_size=15):
    df_to_filter["f_x"] = df_to_filter["x"].rolling(
        window=window_size, min_periods=1).mean()
    df_to_filter["f_y"] = df_to_filter["y"].rolling(
        window=window_size, min_periods=1).mean()
    df_to_filter["f_rot_mag"] = df_to_filter["rmag"].rolling(
        window=window_size, min_periods=1).mean()

    df_rounded = df_to_filter.round(4)
    return df_rounded


if __name__ == "__main__":
    hand_spans, hand_depths = load_measurements()

    folder_path, subject_name, hand = prompts.request_name_hand_simple("csv/")

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
                
                print("FILE: " + str(total_path))

                try:
                    df = pd.read_csv(total_path, 
                        names=["roll", "pitch", "yaw", "x", "y", "z", "tmag", "rmag"])

                except:
                    print("FAILED")
                    continue

                #print(" ")
                #print("CONVERT M TO MM")
                #convert m to mm in translational data
                df = df * [1., 1., 1., 1000., 1000., 1000., 1000., 1.]
                df = df.round(4)

                #print(" ")
                #print("NORMALIZE DATA BY HAND DIMENSION")
                #print(f"Hand span: {hand_spans[hand]}")
                #print(f"Hand depth: {hand_depths[hand]}")
                #normalize translational data by hand span
                df = df / [1., 1., 1., #orientation data
                    hand_spans[hand], #x
                    hand_depths[hand], #y
                    1., #z - doesn't matter
                    1., #translational magnitude - don't use
                    1.]
                df = df.round(4)

                filtered_df = moving_average(df, window_size=15)
                
                #filtered_df.dropna()

                #print("DATA CONDITIONING COMPLETE.")

                #store in a new file
                print("GENERATING FILE FOR: " + file_name)
                new_file_name = "filtered/filt_" + file_name 
                filtered_df.to_csv(new_file_name, index=True, columns = ["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"])
                #print("FILE COMPLETED!")

