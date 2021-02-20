
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv
import pandas as pd
from asterisk_trial import AsteriskTrialData
from asterisk_calculations import Pose2D
import pdb


class AveragedTrial(AsteriskTrialData):

    def __init__(self):
        super(AveragedTrial, self).__init__()  # makes an empty AsteriskTrialData object

        self.names = []  # names of trials averaged
        self.averaged_trials = []  # actual AsteriskTrialData objects that were averaged
        # self.pose_average = []  # maybe just use poses
        self.pose_sd = []

    # def get_poses(self):
    #     """
    #     Separates poses into x, y, theta for easy plotting.
    #     :param: filt_flag Gives option to return filtered or unfiltered data
    #     """
    #     # get the poses
    #     x_data = []
    #     y_data = []
    #     theta_data = []
    #
    #     for pose in self.poses:
    #         x_data.append(pose.x)
    #         y_data.append(pose.y)
    #         theta_data.append(pose.theta)
    #
    #     return x_data, y_data, theta_data

    def get_poses_sd(self):
        """
        Separates poses into x, y, theta for easy plotting.
        :param: filt_flag Gives option to return filtered or unfiltered data
        """
        # get the poses
        x_data = []
        y_data = []
        theta_data = []

        for pose in self.pose_sd:
            x_data.append(pose.x)
            y_data.append(pose.y)
            theta_data.append(pose.theta)

        return x_data, y_data, theta_data

    def get_points(self, x_val, bounds):
        """
        Function which gets all the points that fall in a specific value range
        """
        data_points = pd.DataFrame()  # makes an empty dataframe
        hi_val = x_val + bounds
        lo_val = x_val - bounds

        print(f"t_pose: {x_val} +/- {bounds}")

        for t in self.averaged_trials:
            data_points = data_points.append(t.poses)

        points_in_bounds = data_points[(data_points['x'] > lo_val) & (data_points['x'] < hi_val)]

        # print("selected datapoints")
        # print(points_in_bounds)
        # print("   ")

        return points_in_bounds

    def make_average_line(self, trials):
        """
        Average the path of 2 or more AsteriskTrialObjects
        """

        # collect the asterisktrialdata objects
        self.names = []  # if rerunning an average with same object, make sure these lists are empty
        self.averaged_trials = []
        for t_n in trials:
            self.names.append(t_n.generate_name())
            self.averaged_trials.append(t_n)

        # first take attributes of first asterisktrialdata object and take its attributes
        trial = self.averaged_trials[0]
        self.subject = trial.subject  # TODO: add more subjects, make this a list?
        self.trial_translation = trial.trial_translation
        self.trial_rotation = trial.trial_rotation
        self.trial_num = trial.trial_num

        # TODO: investigate, if trial intersects x axis, trial line will stop there
        self.target_line = trial.target_line

        # TODO: rotate the line so we can do everything based on the x axis?
        # TODO: right now, will just focus on c direction, then expand to the rest. ONLY WORKS FOR C DIRECTION TESTS!!!

        avg_line = pd.DataFrame()
        avg_std = pd.DataFrame()

        # now we go through averaging
        for t in self.target_line:
            t_x = t[0]
            points = self.get_points(t_x, 0.1)
            averaged_point = points.mean(axis=0)  # averages each column in DataFrame
            std_point = points.std(axis=0)
            avg_line = avg_line.append(averaged_point, ignore_index=True)
            avg_std = avg_std.append(std_point, ignore_index=True)

        self.poses = avg_line
        self.pose_sd = avg_std

        # now filter and run fd
        self.moving_average()
        # self.translation_fd, self.rotation_fd = self.calc_frechet_distance()  # TODO: broken, investigate!

        return avg_line


test1 = AsteriskTrialData('sub1_2v2_c_n_1.csv')
test2 = AsteriskTrialData('sub1_2v2_c_n_2.csv')
test3 = AsteriskTrialData('sub1_2v2_c_n_3.csv')

lines = [test1, test2, test3]

avgln = AveragedTrial()
avgln.make_average_line(lines)

# make a comparison plot!
import matplotlib.pyplot as plt
plt.plot(test1.poses['x'], test1.poses['y'], label="test1")
plt.plot(test2.poses['x'], test2.poses['y'], label="test2")
plt.plot(test3.poses['x'], test3.poses['y'], label="test3")
plt.plot(avgln.poses['x'], avgln.poses['y'], label="avg")
plt.plot(avgln.poses['f_x'], avgln.poses['f_y'], label="f_avg")
plt.legend()
plt.show()
