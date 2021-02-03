#!/usr/bin/env python3

import csv
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AsteriskTestMetrics import Pose2D

from asterisk_prompts import hand
from scipy import stats


class AsteriskTrial:
    def __init__(self, file_name):
        """
        Class to represent a single asterisk test trial. Contains:
        :attribute hand - hand object with info for hand involved in the trial (see above)
        :attribute subject_num - integer value for subject number
        :attribute direction - single lettered descriptor for which direction the object travels in for this trial
        :attribute trial_type - indicates one-step or two-step trial as a string (None, Plus15, Minus15)
        :attribute trial_num - integer number of the trial 

        :attribute poses - pandas dataframe containing the object's trajectory (as floats)
        :attribute filtered - boolean that indicates whether trial has been filtered or not
        :attribute ideal_poses - pandas dataframe containing the 'perfect trial' line that we will compare our trial to using Frechet Distance. 
        This 'perfect trial' line is a line that travels in the trial direction (with no deviations) to the max travel distance the 
        trial got to in the respective direction. This is denoted as the projection of the object trajectory on the direction

        :attribute total_distance - float value
        :attribute frechet_distance - float value
        :attribute dist_along_translation - float
        :attribute dist_along_twist - float

        """
        s, h, d, r, e = file_name.split("_")
        n, _ = e.split(".")

        self.hand = h
        self.subject_num = s
        self.trial_translation = d
        self.trial_rotation = r
        self.trial_num = n

        # Data will not be filtered here
        self.poses = self.read_file(file_name)

        self.filtered = False
        self.ideal_poses = None

        self.total_distance = None
        self.frechet_distance = None
        self.dist_along_translation = None
        self.dist_along_twist = None

    def read_file(self, file, folder=""):
        """
        Function to read file and save relevant data in the object
        """
        total_path = folder + file

        try:
            df_temp = pd.read_csv(total_path,
                                  # names=["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"],
                                  skip_blank_lines=True
                                  )

            df = self.condition_df(df_temp)

        except:  # TODO: add more specific except clauses
            df = None
            print(f"{total_path} has failed to read csv")

        return df["x", "y", "rmag"]

    def condition_df(self, df):
        """
        Data conditioning procedure used to:
        0) Make columns of the dataframe numeric (they aren't by default), makes dataframe header after the fact to avoid errors with apply function
        1) convert translational data from meters to mm
        2) normalize translational data by hand span/depth
        3) remove extreme outlier values in data
        """
        df_numeric = df.apply(pd.to_numeric)

        # saving for later: ["row", "x", "y", "rmag", "f_x", "f_y", "f_rot_mag"]
        df_numeric.columns = ["roll", "pitch",
                              "yaw", "x", "y", "z", "tmag",  "rmag"]

        # convert m to mm in translational data
        df = df_numeric * [1., 1., 1., 1000., 1000., 1000., 1000., 1.]
        df = df.round(4)

        # normalize translational data by hand span
        df = df / [1., 1., 1.,  # orientation data
                   self.hand.span,  # x
                   self.hand.depth,  # y
                   1.,  # z - doesn't matter
                   1.,  # translational magnitude - don't use
                   1.]  # rotation magnitude
        df = df.round(4)

        # occasionally get an outlier value (probably from vision algorithm), I filter them out here
        inlier_df = self.remove_outliers(df, ["x", "y", "rmag"])

        return inlier_df

    def generate_name(self):
        """
        Generates the codified name of the trial
        :return: string name of trial
        """
        return f"{self.hand.get_name()}_{self.subject_num}_{self.trial_translation}_" \
               f"{self.trial_rotation}_{self.trial_num}"

    def generate_data_csv(self, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        """
        if file_name_overwrite:
            new_file_name = file_name_overwrite
        else:
            new_file_name = self.generate_name() + ".csv"

        # if data has been filtered, we also want to include that in csv generation,
        # otherwise the filtered columns won't exist
        if self.filtered:
            self.poses.to_csv(new_file_name, index=True, columns=[
                "x", "y", "rmag", "f_x", "f_y", "f_rmag"])
        else:
            self.poses.to_csv(new_file_name, index=True, columns=[
                "x", "y", "rmag"])  # TODO: Should I rename columns?

        print(f"CSV File generated with name: {new_file_name}")

    def remove_outliers(self, df_to_fix, columns):
        """
        Removes extreme outliers from data, in 100% quartile.
        Occasionally this happens in the aruco analyzed data and is a necessary function to run.
        """

        for col in columns:
            # see: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
            # q_low = df_to_fix[col].quantile(0.01)
            q_hi = df_to_fix[col].quantile(0.99)

            df_to_fix = df_to_fix[(df_to_fix[col] < q_hi)]

            # print(col)
            # print(f"q_low: {q_low}")
            # print(f"q_hi: {q_hi}")
            # print(" ")

        return df_to_fix

    def moving_average(self, window_size=15):
        """
        Runs a moving average on the pose data. Saves moving average data into new columns with f_ prefix.
        Overwrites previous moving average calculations.
        """
        self.poses["f_x"] = self.poses["x"].rolling(
            window=window_size, min_periods=1).mean()
        self.poses["f_y"] = self.poses["y"].rolling(
            window=window_size, min_periods=1).mean()
        self.poses["f_rmag"] = self.poses["rmag"].rolling(
            window=window_size, min_periods=1).mean()

        self.poses.round(4)
        self.filtered = True
        print("Moving average completed.")

    def get_pose2d(self):
        """
        Returns the poses for this trial, separately by axis.
        """
        poses = []

        for p in self.poses.iterrows():
            pose = Pose2D(p["x"], p["y"], p["rmag"])
            poses.append(pose)

        return poses # Todo: test this out!

    def get_poses(self, filt_flag=False):
        """
        Separates poses into x, y, theta for easy plotting.
        :param: filt_flag Gives option to return filtered or unfiltered data
        """
        if self.filtered and filt_flag: #flag is there to default to get filtered data if there is filtered data
            x = self.poses["f_x"]
            y = self.poses["f_y"]
            twist = self.poses["f_rmag"]
        else:
            x = self.poses["x"]
            y = self.poses["y"]
            twist = self.poses["rmag"]

        return x, y, twist

    def plot_trial(self, file_name=None):  # TODO: make it so that we can choose filtered or unfiltered data
        """
        Plot the poses in the trial, using marker size to denote the error in twist from the desired twist
        """
        data_x, data_y, theta = self.get_poses()

        plt.plot(data_x, data_y, color='tab:red', label='trajectory')

        # plt.scatter(data_x, data_y, marker='o', color='red', alpha=0.5, s=5*theta)

        # plot data points separately to show angle error with marker size
        for n in range(len(data_x)):
            # TODO: rn having difficulty doing marker size in a batch, so plotting each point separately
            plt.plot(data_x[n], data_y[n], 'r.',
                     alpha=0.5, markersize=5*theta[n])

            max_x = max(data_x)
            max_y = max(data_y)
            min_x = min(data_x)

            # print(f"max_x: {max_x}, min_x: {min_x}, y: {max_y}")

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Path of Object')
            # plt.grid()

            plt.xticks(np.linspace(self.round_half_down(min_x, decimals=2),
                                   self.round_half_up(max_x, decimals=2), 10), rotation=30)
            # gives a realistic view of what the path looks like
            # plt.xticks(np.linspace(0, round_half_up(max_y, decimals=2), 10), rotation=30)
            plt.yticks(np.linspace(0, self.round_half_up(max_y, decimals=2), 10))

            # plt.xlim(0., 0.5)
            # plt.ylim(0., 0.5)

        if file_name:
            plt.savefig(f"plot_{file_name}.jpg", format='jpg')
            plt.show()

    # TODO: is there a better place to put these functions?
    def round_half_up(self, n, decimals=0):
        """
        Used for plotting
        # from: https://realpython.com/python-rounding/
        """
        multiplier = 10 ** decimals
        return m.floor(n*multiplier + 0.5) / multiplier

    def round_half_down(self, n, decimals=0):
        """
        Used for plotting
        # from: https://realpython.com/python-rounding/
        """
        multiplier = 10 ** decimals
        return m.ceil(n*multiplier - 0.5) / multiplier

    def generate_ideal_line(self):
        """
        Using object trajectory (self.poses), build a line to compare to for frechet distance
        """
        pass

    def calculate_frechet_distance(self):
        """
        Calculate the frechet distance between self.poses and self.ideal_line
        """
        pass
