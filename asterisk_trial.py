#!/usr/bin/env python3

import csv
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AsteriskTestMetrics import Pose2D

from asterisk_prompts import hand
from scipy import stats


class ast_trial:
    def __init__(self, file_name):
        '''
        Class to represent a single asterisk test trial. Contains:
        :param hand - hand object with info for hand involved in the trial (see above)
        :param subject_num - integer value for subject number
        :param direction - single lettered descriptor for which direction the object travels in for this trial
        :param trial_type - indicates one-step or two-step trial as a string (None, Plus15, Minus15)
        :param trial_num - integer number of the trial 

        :param poses - pandas dataframe containing the object's trajectory (as floats)
        :param filtered - boolean that indicates whether trial has been filtered or not
        :param ideal_poses - pandas dataframe containing the 'perfect trial' line that we will compare our trial to using Frechet Distance. 
        This 'perfect trial' line is a line that travels in the trial direction (with no deviations) to the max travel distance the 
        trial got to in the respective direction. This is denoted as the projection of the object trajectory on the direction

        :param total_distance - float value
        :param frechet_distance - float value
        :param dist_along_translation - float
        :param dist_along_twist - float

        '''
        s, h, d, r, e = file_name.split("_")
        n, _ = e.split(".")

        self.hand = h
        self.subject_num = s
        self.trial_translation = d
        self.trial_rotation = r
        self.trial_num = n

        # Data will not be filtered here
        self.poses = self.read_file(file_name)
        self.data_conditioning(self.poses) #condition it right away

        self.filtered = False
        self.ideal_poses = None

        self.total_distance = None
        self.frechet_distance = None
        self.dist_along_translation = None
        self.dist_along_twist = None

    def read_file(self, file, folder=""):
        '''
        Function to read file and save relevant data in the object
        '''
        total_path = folder + file

        try:
            df_temp = pd.read_csv(total_path,
                                  #names=["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"],
                                  skip_blank_lines=True
                                  )

            df = self.condition_df(df_temp)

        except:
            df = None
            print(f"{total_path} has failed to read csv")

        return df["x", "y", "rmag"]

    # TODO: is there a better place to put these functions?

    def round_half_up(self, n, decimals=0):
        '''

        #from: https://realpython.com/python-rounding/
        '''
        multiplier = 10 ** decimals
        return m.floor(n*multiplier + 0.5) / multiplier

    def round_half_down(self, n, decimals=0):
        '''

        #from: https://realpython.com/python-rounding/
        '''
        multiplier = 10 ** decimals
        return m.ceil(n*multiplier - 0.5) / multiplier

    def condition_df(self, df):
        '''
        Make columns of the dataframe numeric (they aren't by default)
        Makes dataframe header after the fact to avoid errors with apply function
        '''
        df_numeric = df.apply(pd.to_numeric)

        # saving for later: ["row", "x", "y", "rmag", "f_x", "f_y", "f_rot_mag"]
        df_numeric.columns = ["roll", "pitch",
                              "yaw", "x", "y", "z", "tmag",  "rmag"]

        return df_numeric

    def generate_name(self):
        '''
        Generates the codified name of the trial
        :return:
        '''
        return self.hand.get_name() + "_" + self.subject_num + "_" + self.trial_type + \
            "_" + self.direction + "_" + self.trial_num

    def generate_data_csv(self, file_name_overwrite=None):
        '''
        Saves pose data as a new csv file
        '''
        if(file_name_overwrite):
            new_file_name = file_name_overwrite
        else:
            new_file_name = self.generate_name + ".csv"

        self.poses.to_csv(new_file_name, index=True, columns=[
            "x", "y", "rmag"])  # TODO: Should I rename columns?

    def data_conditioning(self, data, window=15):
        '''
        Data conditioning procedure used to:
        1) convert translational data from meters to mm
        2) normalize translational data by hand span/depth
        3) remove extreme outlier values in data
        4) run a moving average on data
        '''
        # convert m to mm in translational data
        #df = data * [1., 1., 1., 1000., 1000., 1000., 1000., 1.]
        df = data * [1000., 1000., 1.]
        df = df.round(4)

        # normalize translational data by hand span
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
        # TODO: Maybe I should cut out the moving average part of the data conditioning to be another step?
        filtered_df = self.moving_average(inlier_df, window_size=window)

        self.poses = filtered_df
        self.filtered = True
        #print("Data has been conditioned.")

    def remove_outliers(self, df_to_fix, columns):
        '''
        Removes extreme outliers from data, in 100% quartile. Occasionally this happens in the aruco analyzed data
        '''

        for col in columns:
            # see: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
            #q_low = df_to_fix[col].quantile(0.01)
            q_hi = df_to_fix[col].quantile(0.99)

            df_to_fix = df_to_fix[(df_to_fix[col] < q_hi)]

            # print(col)
            #print(f"q_low: {q_low}")
            #print(f"q_hi: {q_hi}")
            #print(" ")

        return df_to_fix

    def moving_average(self, df_to_filter, window_size=15):
        '''
        Runs a moving average on the pose data
        '''
        df_to_filter["f_x"] = df_to_filter["x"].rolling(
            window=window_size, min_periods=1).mean()
        df_to_filter["f_y"] = df_to_filter["y"].rolling(
            window=window_size, min_periods=1).mean()
        df_to_filter["f_rmag"] = df_to_filter["rmag"].rolling(
            window=window_size, min_periods=1).mean()

        df_rounded = df_to_filter.round(4)
        return df_rounded

    def get_pose2d(self):
        '''
        Returns the poses for this trial, separately by axis.
        '''
        #x = self.poses["x"]
        #y = self.poses["y"]
        #twist = self.poses["rmag"]

        poses = []

        for p in self.poses.iterrows():
            pose = Pose2D(p["x"], p["y"], p["rmag"])
            poses.append(pose)

        return poses #todo: test this out!

    def plot_trial(self, file_name=None):
        '''
        Plot the poses in the trial, using marker size to denote the error in twist from the desired twist
        '''
        data_x, data_y, theta = self.get_poses()

        plt.plot(data_x, data_y, color='tab:red', label='trajectory')

        #plt.scatter(data_x, data_y, marker='o', color='red', alpha=0.5, s=5*theta)

        # plot data points separately to show angle error with marker size
        for n in range(len(data_x)):
            # TODO: rn having difficulty doing marker size in a batch, so plotting each point separately
            plt.plot(data_x[n], data_y[n], 'r.',
                     alpha=0.5, markersize=5*theta[n])

            max_x = max(data_x)
            max_y = max(data_y)
            min_x = min(data_x)

            #print(f"max_x: {max_x}, min_x: {min_x}, y: {max_y}")

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Path of Object')
            # plt.grid()

            plt.xticks(np.linspace(self.round_half_down(min_x, decimals=2),
                                   self.round_half_up(max_x, decimals=2), 10), rotation=30)
            # plt.xticks(np.linspace(0, round_half_up(max_y, decimals=2), 10), rotation=30) #gives a realistic view of what the path looks like
            plt.yticks(np.linspace(0, self.round_half_up(max_y, decimals=2), 10))

            #plt.xlim(0., 0.5)
            #plt.ylim(0., 0.5)

        if(file_name):
            plt.savefig("plot4_" + file_name + ".jpg", format='jpg')
            plt.show()

    def generate_ideal_line(self):
        '''
        Using object trajectory (self.poses), build a line to compare to for frechet distance
        '''
        pass

    def calculate_frechet_distance(self):
        '''
        Calculate the frechet distance between self.poses and self.ideal_line
        '''
        pass
