
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv
import pandas as pd
from asterisk_trial import AsteriskTrialData
from asterisk_calculations import Pose2D


class AveragedTrial(AsteriskTrialData):

    def __init__(self):
        super(AveragedTrial, self).__init__()

        self.names = []  # names of trials averaged
        self.averaged_trials = []  # actual AsteriskTrialData objects that were averaged
        self.pose_average = []
        self.pose_sd = []

    def get_poses(self):
        """
        Separates poses into x, y, theta for easy plotting.
        :param: filt_flag Gives option to return filtered or unfiltered data
        """
        # get the poses
        x_data = []
        y_data = []
        theta_data = []

        for pose in self.pose_average:
            x_data.append(pose.x)
            y_data.append(pose.y)
            theta_data.append(pose.theta)

        return x_data, y_data, theta_data

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

    def average_lines(self, trials: [AsteriskTrialData]):
        """ Average the path of 2 or more trials
        :param trials = array of AsteriskTestData objects for one specific trial
        :returns array of average poses with += poses
        """

        # initializing
        self.pose_average = []
        self.pose_sd = []

        self.trial_translation = trials[0].trial_translation
        self.trial_translation = trials[0].trial_rotation

        # This is really clunky, but it's the easiest way to deal
        # with the problem that the arrays have different sizes...
        # n_max = max([len(t.target_indices) for t in trials])
        n_max = max([len(t.translation_indices) for t in trials])

        # make a bunch of empty Pose2D objects - this will be the average line
        self.pose_average = [Pose2D() for _ in range(0, n_max)]
        sd_dist = [0] * n_max
        sd_theta = [0] * n_max
        count = [0] * n_max
        for t in trials:
            # keep track of which trials were averaged here
            self.names.append(t.generate_name())
            self.averaged_trials.append(t)

            for j, index in enumerate(t.translation_indices):  # TODO: do we use rotation indices or don't care?
                print(f"{index} {t.obj_poses[0, index]} {t.obj_poses[1, index]}")
                self.pose_average[j].x += t.obj_poses[0, index]  # TODO: translate t.obj_poses to ... (what is it?)
                self.pose_average[j].y += t.obj_poses[1, index]
                self.pose_average[j].theta += t.obj_poses[2, index]
                count[j] += 1

        # Average
        for i, c in enumerate(count):
            self.pose_average[i].x /= c
            self.pose_average[i].y /= c
            self.pose_average[i].theta /= c
            count[i] = 0

        # SD - do theta separately from distance to centerline
        for t in trials:
            for j, index in enumerate(t.target_indices):
                p = t.obj_poses[:, index]
                dx = self.pose_average[j].x - p[0]  # TODO: translate t.obj_poses to ... (what is it?)
                dy = self.pose_average[j].y - p[1]
                dist = sqrt(dx * dx + dy * dy)
                dt = abs(self.pose_average[j].theta - p[2])
                sd_theta[j] += dt
                sd_dist[j] += dist
                count[j] += 1

        # Normalize SD
        last_valid_i = 0
        for i, p in enumerate(self.pose_average):
            if count[i] > 1:
                self.pose_sd.append((sd_dist[i] / (count[i] - 1), sd_theta[i] / (count[i] - 1)))
                last_valid_i = i
            else:
                self.pose_sd.append(self.pose_sd[last_valid_i])

        # no return, just use the object itself