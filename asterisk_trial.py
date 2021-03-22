#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_calculations import AsteriskCalculations
import similaritymeasures as sm
from asterisk_plotting import AsteriskPlotting as aplt
from asterisk_metrics import AsteriskMetrics as am
import pdb
from asterisk_hand import HandObj
from scipy import stats


class AsteriskTrialData:
    def __init__(self, file_name=None, do_fd=True):
        """
        Class to represent a single asterisk test trial.
        :param file_name: - name of the file that you want to import data from

        Class contains:
        :attribute hand: - hand object with info for hand involved in the trial (see above)
        :attribute subject_num: - integer value for subject number
        :attribute direction: - single lettered descriptor for which direction the object travels in for this trial
        :attribute trial_type: - indicates one-step or two-step trial as a string (None, Plus15, Minus15)
        :attribute trial_num: - integer number of the trial

        :attribute poses: - pandas dataframe containing the object's trajectory (as floats)
        :attribute filtered: - boolean that indicates whether trial has been filtered or not
        :attribute ideal_poses: - pandas dataframe containing the 'perfect trial' line that we will compare our trial to using Frechet Distance.
        This 'perfect trial' line is a line that travels in the trial direction (with no deviations) to the max travel distance the 
        trial got to in the respective direction. This is denoted as the projection of the object trajectory on the direction

        :attribute total_distance: - float value
        :attribute frechet_distance: - float value
        :attribute dist_along_translation: - float
        :attribute dist_along_twist: - float
        """
        if file_name:
            s, h, t, r, e = file_name.split("_")
            n, _ = e.split(".")
            self.hand = HandObj(h)

            # Data will not be filtered in this step
            data = self._read_file(file_name)
            self.poses = data[["x", "y", "rmag"]]

        else:
            s, t, r, n = None, None, None, None
            self.hand = None

        self.subject = s
        self.trial_translation = t
        self.trial_rotation = r
        self.trial_num = n

        if file_name:
            print(self.generate_name())

        self.filtered = False
        self.window_size = 0

        self.target_line = None  # the straight path in the direction that this trial is
        self.target_rotation = None

        # metrics
        self.total_distance = None
        self.translation_fd = None
        self.rotation_fd = None
        self.mvt_efficiency = None
        self.area_btwn = None

        if file_name:
            self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
            self.target_rotation = self.generate_target_rot()  # TODO: doesn't work for true cw and ccw yet

            if do_fd:
                self.update_all_metrics()

    def add_hand(self, hand_name):
        """
        If you didn't make the object with a file_name, a function to set hand in painless manner
        :param hand_name: name of hand to make
        """
        self.hand = HandObj(hand_name)

    def _read_file(self, file_name, folder="csv/"):
        """
        Function to read file and save relevant data in the object
        :param file_name: name of file to read in
        :param folder: name of folder to read file from. Defaults csv folder
        """
        total_path = f"{folder}{file_name}"
        try:
            df_temp = pd.read_csv(total_path,
                                  # names=["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"],
                                  skip_blank_lines=True
                                  )
            df = self._condition_df(df_temp)

        except Exception as e:  # TODO: add more specific except clauses
            # print(e)
            df = None
            # print(f"{total_path} has failed to read csv")
        return df

    def _condition_df(self, df):
        """
        Data conditioning procedure used to:
        0) Make columns of the dataframe numeric (they aren't by default), makes dataframe header after the fact to avoid errors with apply function
        1) convert translational data from meters to mm
        2) normalize translational data by hand span/depth
        3) remove extreme outlier values in data
        """
        df_numeric = df.apply(pd.to_numeric)

        # saving for later: ["row", "x", "y", "rmag", "f_x", "f_y", "f_rot_mag"]
        df_numeric.columns = ["roll", "pitch", "yaw", "x", "y", "z", "tmag",  "rmag"]

        # convert m to mm in translational data
        df = df_numeric * [1., 1., 1., 1000., 1000., 1000., 1000., 1.]
        df.round(4)

        # normalize translational data by hand span
        df = df / [1., 1., 1.,  # orientation data
                   self.hand.span,  # x
                   self.hand.depth,  # y
                   1.,  # z - doesn't matter
                   1.,  # translational magnitude - don't use
                   1.]  # rotation magnitude
        df.round(4)

        # occasionally get an outlier value (probably from vision algorithm), I filter them out here
        inlier_df = self._remove_outliers(df, ["x", "y", "rmag"])
        return inlier_df.round(4)

    def is_trial(self, subject_name, hand_name, translation_name, rotation_name, trial_num=None):
        """  TODO: not tested yet
        a function that returns whether this trial is equivalent to the parameters listed
        :param subject_name: name of subject
        :param hand_name: name of hand
        :param translation_name: name of translation trial
        :param rotation_name: name of rotation trial
        :param trial_num: trial number, default parameter
        """
        # TODO: make with *args instead, that way we can specify as much as we want to
        if subject_name == self.subject and hand_name == self.hand.get_name() \
                and translation_name == self.trial_translation \
                and rotation_name == self.trial_rotation:
            if trial_num and trial_num == self.trial_num:
                return True
            elif trial_num:
                return False
            else:
                return True
        else:
            return False

    def generate_name(self):
        """
        Generates the codified name of the trial
        :return: string name of trial
        """
        return f"{self.subject}_{self.hand.get_name()}_{self.trial_translation}_" \
               f"{self.trial_rotation}_{self.trial_num}"

    def save_data(self, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        :param file_name_overwrite: optional parameter, will save as generate_name unless a different name is specified
        """
        if file_name_overwrite is None:
            new_file_name = self.generate_name() + ".csv"

        else:
            new_file_name = file_name_overwrite + ".csv"

        # if data has been filtered, we also want to include that in csv generation,
        # otherwise the filtered columns won't exist
        if self.filtered:  # TODO: make it in a special folder?
            filtered_file_name = f"filtered/f{self.window_size}_{new_file_name}"

            self.poses.to_csv(filtered_file_name, index=True, columns=[
                "x", "y", "rmag", "f_x", "f_y", "f_rmag"])
        else:
            self.poses.to_csv(new_file_name, index=True, columns=[
                "x", "y", "rmag"])

        # print(f"CSV File generated with name: {new_file_name}")

    def _remove_outliers(self, df_to_fix, columns):
        """
        Removes extreme outliers from data, in 99% quartile.
        Occasionally this happens in the aruco analyzed data and is a necessary function to run.
        These values completely mess up the moving average filter unless they are dealt with earlier.
        :param df_to_fix: the dataframe to fix
        :param columns: dataframe columns to remove outliers from
        """
        for col in columns:
            # see: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
            # q_low = df_to_fix[col].quantile(0.01)
            q_hi = df_to_fix[col].quantile(0.99)

            df_to_fix = df_to_fix[(df_to_fix[col] < q_hi)]  # this has got to be the problem line

            # print(col)
            # print(f"q_low: {q_low}")
            # print(f"q_hi: {q_hi}")
            # print(" ")

        return df_to_fix

    def moving_average(self, window_size=15):
        """
        Runs a moving average on the pose data. Saves moving average data into new columns with f_ prefix.
        Overwrites previous moving average calculations.
        :param window_size: size of moving average. Defaults to 15.
        """
        # TODO: makes a bunch of nan values at end of data
        self.poses["f_x"] = self.poses["x"].rolling(
            window=window_size, min_periods=1).mean()
        self.poses["f_y"] = self.poses["y"].rolling(
            window=window_size, min_periods=1).mean()
        self.poses["f_rmag"] = self.poses["rmag"].rolling(
            window=window_size, min_periods=1).mean()

        self.poses = self.poses.round(4)
        self.filtered = True
        self.window_size = window_size

    def get_poses(self, use_filtered=True):
        """
        Separates poses into x, y, theta for easy plotting.
        :param: use_filtered: Gives option to return filtered or unfiltered data
        """
        if self.filtered and use_filtered:
            x = self.poses["f_x"]
            y = self.poses["f_y"]
            twist = self.poses["f_rmag"]
        else:
            x = self.poses["x"]
            y = self.poses["y"]
            twist = self.poses["rmag"]

        return_x = pd.Series.to_list(x.dropna())
        return_y = pd.Series.to_list(y.dropna())
        return_twist = pd.Series.to_list(twist.dropna())

        return return_x, return_y, return_twist

    def plot_trial(self, use_filtered=True, show_plot=True, save_plot=False):
        """
        Plot the poses in the trial, using marker size to denote the error in twist from the desired twist
        :param use_filtered: Gives option to return filtered or unfiltered data
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        data_x, data_y, theta = self.get_poses(use_filtered)

        # experimenting...
        # junk = 70
        # data_x = data_x[0:(len(data_x)-junk)]
        # data_y = data_y[0:(len(data_y)-junk)]

        plt.plot(data_x, data_y, color='black', label='trajectory')

        # plot data points separately to show angle error with marker size
        for n in range(len(data_x)):
            # TODO: rn having difficulty doing marker size in a batch, so plotting each point separately
            plt.plot(data_x[n], data_y[n], 'r.',
                     alpha=0.5, markersize=5*theta[n])

        max_x = max(data_x)
        max_y = max(data_y)
        min_x = min(data_x)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path of Object')

        # gives a realistic view of what the path looks like
        plt.xticks(np.linspace(aplt.round_half_down(min_x, decimals=2),
                              aplt.round_half_up(max_x, decimals=2), 10), rotation=30)
        plt.yticks(np.linspace(0, aplt.round_half_up(max_y, decimals=2), 10))

        # experimenting...
        # plt.xticks(np.linspace(-10,
        #                        80, 10), rotation=30)
        # plt.yticks(np.linspace(-10, 80, 10))
        plt.gca().set_aspect('equal', adjustable='box')

        plt.title(f"Plot: {self.generate_name()}")

        if save_plot:
            plt.savefig(f"pics/plot_{self.generate_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def get_last_pose(self):
        """
        Returns last pose as an array. Returns both filtered and unfiltered data if obj is filtered
        """
        return self.poses.tail(1).to_numpy()[0]

    def generate_target_line(self, n_samples=100):
        """
        Using object trajectory (self.poses), build a line to compare to for frechet distance.
        Updates this attribute on object.
        :param n_samples: number of samples for target line. Defaults to 100
        """
        x_vals, y_vals = aplt.get_direction(self.trial_translation, n_samples)

        target_line = np.column_stack((x_vals, y_vals))

        # get last object pose and use it for determining how far target line should go
        last_obj_pose = self.poses.tail(1).to_numpy()[0]

        target_line_length = AsteriskCalculations.narrow_target(last_obj_pose, target_line)

        if target_line_length:
            distance_travelled = AsteriskCalculations.t_distance([0, 0], target_line[target_line_length+1])
            final_target_ln = target_line[:target_line_length]
        else:
            distance_travelled = AsteriskCalculations.t_distance([0, 0], target_line[0])
            final_target_ln = target_line[:1]

        # TODO: distance travelled has error because it is built of target line... maybe use last_obj_pose instead?
        return final_target_ln, distance_travelled

    def generate_target_rot(self, n_samples=50):
        """
        get target rotation to compare to with fd
        :param n_samples: number of samples for target line. TODO: Currently not used
        """
        if self.trial_rotation in ["cw", "ccw"]:
            if self.filtered:
                last_rot = self.poses.tail(1)["f_rmag"]
            else:
                last_rot = self.poses.tail(1)["rmag"]

            target_rot = pd.Series.to_list(last_rot)

        # TODO: we compute rotation magnitude, so no neg values ever show up, revisit how rotation is calc'd?
        # elif self.trial_rotation == "ccw":
        #     last_rot = self.poses.tail["rmag"]
        #     target_rot = np.array([-last_rot])

        elif self.trial_rotation in ["p15", "m15"]:
            target_rot = np.array([15])

        # elif self.trial_rotation == "m15":
        #     target_rot = np.array([-15])

        else:
            target_rot = np.zeros(1)

        return target_rot

    def calc_rot_err(self, use_filtered=True):
        """
        calculate and return the error in rotation for every data point
        :param: use_filtered: Gives option to return filtered or unfiltered data
        """

        if self.filtered and use_filtered:
            rots = self.poses["f_rmag"]
        else:
            rots = self.poses["rmag"]

        # subtract desired rotation
        rots = rots - self.target_rotation

        return pd.Series.to_list(rots)

    def update_all_metrics(self, use_filtered=True):
        """
        Updates all metric values on the object
        """
        self.translation_fd, self.rotation_fd = am.calc_frechet_distance(self)
        self.mvt_efficiency = am.calc_mvt_efficiency(self)
        self.area_btwn = am.calc_area_btwn_curves(self)

        # TODO: return anything?


if __name__ == '__main__':
    test = AsteriskTrialData("sub1_2v2_c_n_1.csv")
    test.plot_trial(use_filtered=False)
