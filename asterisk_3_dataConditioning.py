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

from scipy import stats

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

def remove_outliers(df_to_fix, columns):
    #df_filtered = df_to_fix

    for col in columns:
        #see: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
        #q_low = df_to_fix[col].quantile(0.01)
        q_hi  = df_to_fix[col].quantile(0.99)

        df_to_fix = df_to_fix[(df_to_fix[col] < q_hi)]

        #print(col)
        #print(f"q_low: {q_low}")
        #print(f"q_hi: {q_hi}")
        #print(" ")

    return df_to_fix


def moving_average(df_to_filter, window_size=15):
    df_to_filter["f_x"] = df_to_filter["x"].rolling(
        window=window_size, min_periods=1).mean()
    df_to_filter["f_y"] = df_to_filter["y"].rolling(
        window=window_size, min_periods=1).mean()
    df_to_filter["f_rot_mag"] = df_to_filter["rmag"].rolling(
        window=window_size, min_periods=1).mean()

    df_rounded = df_to_filter.round(4)
    return df_rounded

def average_the_lines(dfs, line_size):
    #see: https://stackoverflow.com/questions/49037902/how-to-interpolate-a-line-between-two-other-lines-in-python 
    #or: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html 
    resampled_dfs = list()

    #resampled the paths so they are all the same length
    for df in dfs:
        #new_df = resample_paths(samples=line_size) #TODO:not done here
        #resampled_dfs.append(new_df)

    #interpolate the average between each point... also get the std at each point
    average_line = list()

    for i in range(line_size):
        #get average
        averaged_point = get_average_point(resampled_dfs, i) #TODO: Include std in the output here

        #get std
        #point_std

        average_line.append(averaged_point)


    #plot the average line

def get_average_point(dfs, index):

    #get average x at index


    #get average y at index


    #get average theta at index



     

if __name__ == "__main__":
    hand_spans, hand_depths = load_measurements()

    folder_path, subject_name, hand = prompts.request_name_hand_simple("csv/")

    #num = "3"

    #file_name = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num + ".csv"

    #total_path = folder_path + file_name

    if hand == "basic" or hand == "m2stiff" or hand == "modelvf":
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
                    print(f"FAILED: {total_path}")
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

                inlier_df = remove_outliers(df, ["x", "y", "rmag"]) #occasionally get a val waaaay above 1, I filter them out here
                filtered_df = moving_average(inlier_df, window_size=15) #TODO: Maybe I should do teh translational data normalization after the filtering?
                
                #filtered_df.dropna()

                #print("DATA CONDITIONING COMPLETE.")

                #store in a new file
                print("GENERATING FILE FOR: " + file_name)
                new_file_name = "filtered/filt_" + file_name 
                filtered_df.to_csv(new_file_name, index=True, columns = ["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"])
                #print("FILE COMPLETED!")

