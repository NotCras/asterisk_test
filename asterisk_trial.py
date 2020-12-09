#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
from scipy import stats



class hand:

    def __init__(self, name, fingers):
        spans, depths = self.load_measurements()

        self.hand_name = name
        self.span = spans[name]
        self.depth = depths[name]
        self.num_fingers = fingers

    def load_measurements(self):
        #import hand span data
        spans = dict()
        depths = dict()

        print("LOADING HAND MEASUREMENTS")
        with open(".hand_dimensions") as csv_file:
            csv_reader_hands = csv.reader(csv_file, delimiter=',')
            #populating dictionaries with dimensions
            for row in csv_reader_hands:
                #TODO: make it so that we just return the hand span and depth that we need
                hand_name = row[0]
                hand_span = float(row[1])
                hand_depth = float(row[2])

                spans[hand_name] = hand_span
                depths[hand_name] = hand_depth
        
        return spans, depths



class ast_trial:

    def __init__(self, file_name):
        #TODO: Check the order of the entries
        h, s, t, d, e = file_name.split("_")
        n, _ = e.split(".")

        self.hand = h
        self.subject_num = s
        self.direction = d
        self.trial_type = t
        self.trial_num = n

        df_poses = self.read_file(file_name)

        self.poses = df_poses["x", "y", "rmag"] #Data will not be filtered here
        self.filtered = False
        self.ideal_poses = None

        self.total_distance = None
        self.frechet_distance = None
        self.dist_along_translation = None
        self.dist_along_twist = None

    def read_file(self, file, folder=""):

        total_path = folder + file
        try:
            df = pd.read_csv(total_path,
                names=["roll", "pitch", "yaw", "x", "y", "z", "tmag", "rmag"])

        except:
            self.ideal_poses = None

    def generate_file(self):
        new_file_name = " "

        self.poses.to_csv(new_file_name, index=True, columns=[
                           "x", "y", "rmag"]) #TODO: Should I rename columns?

    def data_conditioning(self, data, window=15):
        #convert m to mm in translational data
        df = data * [1., 1., 1., 1000., 1000., 1000., 1000., 1.]
        df = df.round(4)

        #normalize translational data by hand span
        df = df / [1., 1., 1.,  # orientation data
                   self.hand.span,  # x
                   self.hand.depth,  # y
                   1.,  # z - doesn't matter
                   1.,  # translational magnitude - don't use
                   1.]
        df = df.round(4)

        # occasionally get an outlier value (probably from vision algorithm), I filter them out here
        inlier_df = self.remove_outliers(df, ["x", "y", "rmag"])

        # TODO: Maybe I should do the translational data normalization after the filtering?
        filtered_df = self.moving_average(inlier_df, window_size=window)

        return filtered_df


self.ideal_poses = None

            #print(col)
            #print(f"q_low: {q_low}")
            #print(f"q_hi: {q_hi}")
            #print(" ")

        return df_to_fix


    def moving_average(df_to_filter, window_size=15):
        #TODO: fix errors below
        df_to_filter["f_x"] = df_to_filter["x"].rolling(
            window=window_size, min_periods=1).mean()
        df_to_filter["f_y"] = df_to_filter["y"].rolling(
            window=window_size, min_periods=1).mean()
        df_to_filter["f_rot_mag"] = df_to_filter["rmag"].rolling(
            window=window_size, min_periods=1).mean()

        df_rounded = df_to_filter.round(4)
        return df_rounded

    def plot_trial():
        pass

    def generate_ideal_line():
        pass

    def calculate_frechet_distance():
        pass





