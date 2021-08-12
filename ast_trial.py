#!/usr/bin/env python3

"""
Handles a single asterisk trial. Manages reading in, conditioning, and filtering object paths. Also handles plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_plotting import AsteriskPlotting as aplt
from data_calculations import AsteriskCalculations as acalc
from metric_calculation import AstMetrics as am
from trial_labelling import AsteriskLabelling as al
import pdb
from ast_hand_info import HandInfo
from scipy import stats


class AstTrial:
    def __init__(self, data=None, file_name=None, folder=None, do_metrics=True, norm_data=True,
                 controller_label=None, condition_data=True):
        """ # TODO: comment for constructor
        Class to represent a single asterisk test trial.
        :param data: - name of the file that you want to import data from

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

        self.data_labels = []

        try:
            # Data can either be a dataframe or a filename
            # if a dataframe is provided, add that data
            try:
                if data is not None and isinstance(data, pd.DataFrame):
                    self.add_data_by_df(data, condition_df=condition_data,
                                        do_metrics=do_metrics, norm_data=norm_data)
                    # will need to add demographic data separately
            except Exception as e:
                raise IndexError("Dataframe is not formed correctly.")

            # if a filename is provided and we don't have dataframe data
            if file_name is not None and data is None and isinstance(file_name, str):

                self.demographics_from_filename(filename=file_name)

                self.add_data_by_file(file_name, condition_data=condition_data,
                                      do_metrics=do_metrics, norm_data=norm_data)

            # if a filename is provided and a dataframe is provided
            elif file_name is not None and data is not None and isinstance(file_name, str):
                self.demographics_from_filename(filename=file_name)

            else:
                raise TypeError("Filename failed.")

        except Exception as e:
            print(e)
            print("AstTrial loading failed.")

        # if file_name:
        #     print(self.generate_name())
        self.controller_label = controller_label  # TODO: integrate controller label into the plot title

        self.filtered = False
        self.window_size = 0

    def demographics_from_filename(self, file_name):
        """
        Takes a file_name and gets relevant information out
        """
        s, h, t, r, e = file_name.split("_")
        n, _ = e.split(".")

        self.add_hand_info(h)
        self.data_demographics(s, t, r, n)

    def data_demographics(self, subject=None, translation=None, rotation=None, number=None):
        """
        Add demographics to your data
        """
        self.subject = subject
        self.trial_translation = translation
        self.trial_rotation = rotation
        self.trial_num = number

    def add_hand_info(self, hand_name):
        """
        If you didn't make the object with a file_name, a function to set hand in painless manner
        :param hand_name: name of hand to make
        """
        self.hand = HandInfo(hand_name)

    def assess_data_labels(self, no_mvt_threshold=0.1, init_threshold=0.05, init_num_pts=10):
        """
        Assesses the labels on the data, adds labels to data_labels.
        """
        # check whether total distance is an acceptable distance to consider it actually movement
        if self.total_distance < no_mvt_threshold:
            self.data_labels.append("no_mvt")
            print(f"No movement detected in {self.generate_name()}. Skipping metric calculation.")

        # check that data starts near center
        if not al.assess_initial_position(threshold=init_threshold, to_check=init_num_pts):
            self.data_labels.append("not centered")
            print(f"Data for {self.generate_name()} failed, did not start at center.")

        deviated, dev_perc = al.assess_path_deviation(self)

        if deviated and dev_perc > 0.2:
            self.data_labels.append("major deviation")
            print(f"Detected major deviation in {self.generate_name()}. Labelled trial.")
        elif deviated:
            self.data_labels.append("deviation")
            print(f"Detected minor deviation in {self.generate_name()}. Labelled trial.")

        # TODO: backtracking and shuttling

        return self.data_labels

    def add_data_by_file(self, file_name, norm_data=True, handinfo_name=None, do_metrics=True, condition_data=True):
        """
        Add object path data as a file. By default, will run data through conditioning function
        """
        # TODO: add translation, rotation, and other labels
        self.hand = HandInfo(handinfo_name)

        # Data will not be filtered in this step
        data = self._read_file(file_name, norm_data=norm_data)
        self.poses = data[["x", "y", "rmag"]]

        self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
        self.target_rotation = self.generate_target_rot()  # TODO: doesn't work for true cw and ccw yet

        self.assess_data_labels()

        if do_metrics and self.poses is not None and "no_mvt" not in self.data_labels:
            self.update_all_metrics()

    def add_data_by_df(self, path_df, condition_df=True, do_metrics=True, norm_data=True):
        """
        Add object path data as a dataframe. By default, will run dataframe through conditioning function
        """
        # TODO: add translation, rotation, and other labels
        if condition_df:
            data = self._condition_df(path_df, norm_data)
        else:
            data=path_df

        self.poses = data[["x", "y", "rmag"]]

        self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
        self.target_rotation = self.generate_target_rot()  # TODO: doesn't work for true cw and ccw yet

        self.assess_data_labels()

        if do_metrics and self.poses is not None and "no_mvt" not in self.data_labels:
            self.update_all_metrics()

    def _read_file(self, file_name, folder="aruco_data/", norm_data=True, condition_data=True):
        """
        Function to read file and save relevant data in the object
        :param file_name: name of file to read in
        :param folder: name of folder to read file from. Defaults csv folder
        """
        total_path = f"{folder}{file_name}"
        try:
            # print(f"Reading file: {total_path}")
            df = pd.read_csv(total_path, skip_blank_lines=True)
        except Exception as e:  # TODO: add more specific except clauses
            # print(e)
            print(f"{total_path} has failed to read csv")
            return None

        if condition_data:
            try:
                # print(f"Now at data conditioning.")
                df = self._condition_df(df, norm_data=norm_data)
            except Exception as e:
                # print(e)
                print(f"{total_path} has failed at data conditioning. There's a problem with the data.")
                return None

        return df

    def _condition_df(self, df, norm_data=True):
        """
        Data conditioning procedure used to:
        0) Make columns of the dataframe numeric (they aren't by default), makes dataframe header after the fact to avoid errors with apply function
        1) convert translational data from meters to mm
        2) normalize translational data by hand span/depth
        3) remove extreme outlier values in data
        """
        # df_numeric = df.apply(pd.to_numeric)
        df = df.set_index("frame")

        # df_numeric.columns = ["pitch", "rmag", "roll", "tmag", "x", "y", "yaw", "z"]
        # TODO: is there a way I can make this directly hit each column without worrying about the order?
        # convert m to mm in translational data
        df = df * [1., 1., 1., 1000., 1000., 1000., 1., 1000.]

        if norm_data:
            # normalize translational data by hand span
            df = df / [1., 1., 1.,  # orientation data
                       1., # translational magnitude, don't use
                       self.hand.span,  # x
                       self.hand.depth,  # y
                       1.,  # yaw
                       1.]  # z - doesn't matter
            df = df.round(4)

        # occasionally get an outlier value (probably from vision algorithm), I filter them out here
        #inlier_df = self._remove_outliers(df, ["x", "y", "rmag"])
        if len(df) > 10:  # for some trials with movement, this destroys the data. 10 is arbitrary value that works
            for col in ["x", "y", "rmag"]:
                # see: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
                # q_low = df_to_fix[col].quantile(0.01)
                q_hi = df[col].quantile(0.98)  # determined empirically

                df = df[(df[col] < q_hi)]  # this has got to be the problem line

        return df.round(4)

    def is_ast_trial(self):
        return isinstance(self, AstTrial)

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

    def crop_data(self, start_i=1, end_i=None):
        """
        Enables you to crop data
        """
        if end_i is None:
            end_i = self.poses.shape[0] - 1

        self.poses = self.poses.loc[start_i:end_i]

    def save_data(self, folder=None, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        :param file_name_overwrite: optional parameter, will save as generate_name unless a different name is specified
        """ # TODO: add ability to specific your own folder

        if folder is None:
            folder = "trial_paths/"

        if file_name_overwrite is None:
            new_file_name = f"{self.generate_name()}.csv"

        else:
            new_file_name = f"{file_name_overwrite}.csv"

        # if data has been filtered, we also want to include that in csv generation,
        # otherwise the filtered columns won't exist
        if self.filtered:
            filtered_file_name = f"{folder}f{self.window_size}_{new_file_name}"

            self.poses.to_csv(filtered_file_name, index=True, columns=[
                "x", "y", "rmag", "f_x", "f_y", "f_rmag"])
        else:
            self.poses.to_csv(f"{folder}{new_file_name}", index=True, columns=[
                "x", "y", "rmag"])

        # print(f"CSV File generated with name: {new_file_name}")

    def moving_average(self, window_size=15):
        """
        Runs a moving average on the pose data. Saves moving average data into new columns with f_ prefix.
        Overwrites previous moving average calculations.
        :param window_size: size of moving average. Defaults to 15.
        """
        # TODO: makes a bunch of nan values at end of data... how to fix?
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

    def iterate_poses(self, use_filtered=True):
        """
        Generator to go through each pose in order
        """
        # TODO: need to refactor code to use this
        # get the data you want
        if self.filtered and use_filtered:
            x_list, y_list, t_list = self.get_poses(use_filtered=True)
        else:
            x_list, y_list, t_list = self.get_poses(use_filtered=False)

        # iterate through it!
        for x_val, y_val, t_val in zip(x_list, y_list, t_list):
            yield x_val, y_val, t_val  # TODO: horribly clunky, redo more elegant when I have the chance

    def plot_trial(self, use_filtered=True, show_plot=True, save_plot=False, provide_notes=False, angle_interval=None):
        """
        Plot the poses in the trial, using marker size to denote the error in twist from the desired twist
        :param use_filtered: Gives option to return filtered or unfiltered data
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        data_x, data_y, theta = self.get_poses(use_filtered)

        plt.plot(data_x, data_y, color="xkcd:dark blue", label='trajectory')

        # plot data points separately to show angle error with marker size
        # for n in range(len(data_x)):
        #     # TODO: rn having difficulty doing marker size in a batch, so plotting each point separately
        #     plt.plot(data_x[n], data_y[n], 'r.',
        #              alpha=0.5, markersize=5*theta[n])
        self.plot_orientations(scale=1.0)

        target_x, target_y = [], []
        for t in self.target_line:
            target_x.append(t[0])
            target_y.append(t[1])

        #target_x, target_y = aplt.get_direction(self.trial_translation)
        plt.plot(target_x, target_y, color="xkcd:pastel blue", label="target_line", linestyle="-")

        max_x = max(data_x)
        max_y = max(data_y)
        min_x = min(data_x)
        min_y = min(data_y)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path of Object')

        # gives a realistic view of what the path looks like
        plt.xticks(np.linspace(aplt.round_half_down(min_x, decimals=2),
                              aplt.round_half_up(max_x, decimals=2), 10), rotation=30)
        if self.trial_translation in ["a", "b", "c", "g", "h"]:
            plt.yticks(np.linspace(0, aplt.round_half_up(max_y, decimals=2), 10))
        else:
            plt.yticks(np.linspace(aplt.round_half_down(min_y, decimals=2), 0, 10))

        # plt.gca().set_aspect('equal', adjustable='box')

        plt.title(f"Plot: {self.generate_name()}")

        if provide_notes:
            # TODO: provide notes on the plot in the bottom left corner including labels...
            self.plot_notes()

        if save_plot:
            plt.savefig(f"results/pics/plot_{self.generate_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_notes(self):
        """
        Plots the labels and trial ID in the bottom left corner of the plot
        """ # TODO: implement this!
        note = "Labels: "
        for l in self.data_labels:
            note = f"{note} {l}"

        # TODO: need to check I got the right coordinates
        # TODO: get ax
        # plt.text(0.1, 0.2, self.generate_name(), transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))
        # plt.text(0.1, 0.1, note, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))

    def plot_orientations(self, marker_scale=25, line_length=0.01, positions=[0.3, 0.55, 0.75], scale=0.25):
        """
        Makes a dial at points indicating the current rotation at that point.
        It won't do it for every point, that is indicated in positions.
        A straight up line indicates no rotation.
        Default values are tuned for a single line plot
        :param scale:
        :return:
        """
        # TODO: make positions not mutable so the error stops annoying me
        marker_size = str(int(marker_scale*scale))
        x_data, y_data, t_data = self.get_poses()
        size_data = len(x_data)

        dox = 0.
        doy = scale * 0.01

        if positions is not None:
            for spot in positions:
                idx = int(spot * size_data)
                x = x_data[idx]
                y = y_data[idx]
                t = t_data[idx] * 2 # multiply by 2 to make it on a scale of 180
                print(t)

                plt.plot(x, y, marker="s", markersize=marker_size, color="xkcd:slate", alpha=0.7)
                # makes a custom, square marker with rotation built in
                # plt.plot(x, y, marker=(4, 0, t), markersize=marker_size, color="xkcd:slate", alpha=0.7)
                dx = scale * line_length * np.sin(np.pi*t/180.)
                dy = scale * line_length * np.cos(np.pi*t/180.)

                # plt.plot([x, x + dox], [y, y + doy], linewidth=1, color="xkcd:cream")
                # plt.plot(x, y+doy, markersize=line_length, color="xkcd:aqua green")
                plt.plot([x, x + dx], [y, y + dy], linewidth=1, color="xkcd:cream")
                # plt.pie([t, 180-y], center=[x,y], radius=0.005)

        else:
            # go through each point
            for x, y, t in zip(x_data, y_data, t_data):
                plt.plot(x, y, marker="s", markersize=marker_size, color="xkcd:slate", alpha=0.7)
                dx = scale * line_length * np.cos(np.pi * t / 180.)
                dy = scale * line_length * np.sin(np.pi * t / 180.)

                # plt.plot([x, x + dox], [y, y + doy], linewidth=1, color="xkcd:cream")
                # plt.plot(x, y+doy, markersize=1, color="xkcd:aqua green")
                plt.plot([x, x + dx], [y, y + dy], linewidth=1, color="xkcd:cream")

                # poly = [[x, y], [x + dox, y + doy], [x + dx, y + dy]]
                # polyg = plt.Polygon(poly, color="xkcd:cream", alpha=0.9)
                # plt.gca().add_patch(polyg)

        # always includes the last point
        x = x_data[size_data-1]
        y = y_data[size_data-1]
        t = t_data[size_data-1] * 2
        #print(f"{x}, {y}, r {t}")

        plt.plot(x, y, marker="s", markersize=marker_size, color="xkcd:slate", alpha=0.7)
        dx = scale * line_length * np.sin(np.pi * t / 180.)
        dy = scale * line_length * np.cos(np.pi * t / 180.)

        # plt.plot([x, x + dox], [y, y + doy], linewidth=1, color="xkcd:cream")
        # plt.plot(x, y + doy, markersize=1, color="xkcd:aqua green")
        plt.plot([x, x + dx], [y, y + dy], linewidth=1, color="xkcd:cream")
        # plt.pie([t, 180-y], center=[x, y], radius=0.005)

    def get_last_pose(self):
        """
        Returns last pose as an array. Returns both filtered and unfiltered data if obj is filtered
        """
        return self.poses.dropna().tail(1).to_numpy()[0]

    def check_labels(self, labels_2_check):
        """
        Check whether an asterisk trial object has specific labels. Will tell you which labels were triggered
        """
        result = False
        triggered_labels = []

        if self.data_labels:
            for label in self.data_labels:  # TODO: there's definitely a more elegant way to do this
                if label in labels_2_check:
                    result = True
                    triggered_labels.append(label)

        else:  # there are no labels on the object
            pass

        return result, triggered_labels

    def generate_target_line(self, n_samples=100, no_norm=0):
        """
        Using object trajectory (self.poses), build a line to compare to for frechet distance.
        Updates this attribute on object.
        :param n_samples: number of samples for target line. Defaults to 100
        """

        #if no_norm==0:
        x_vals, y_vals = aplt.get_direction(self.trial_translation, n_samples=n_samples)
        #else:
        #    x_vals, y_vals = aplt.get_direction(self.trial_translation, n_samples=n_samples, max=no_norm)

        target_line = np.column_stack((x_vals, y_vals))

        # get last object pose and use it for determining how far target line should go

        last_obj_pose = self.poses.tail(1).to_numpy()[0]

        target_line_length = acalc.narrow_target(last_obj_pose, target_line)

        if target_line_length <= n_samples - 2:  # want the plus 1 to round up
            distance_travelled = acalc.t_distance([0, 0], target_line[target_line_length + 1])
            final_target_ln = target_line[:target_line_length]

        elif target_line_length == n_samples - 1:   # if at the top,
            distance_travelled = acalc.t_distance([0, 0], target_line[target_line_length])  # + 1])
            final_target_ln = target_line[:target_line_length]

        else:  # covering for no movement
            # unfortunately,  we register a very small translation here, but this is only in case no_mvt fails
            distance_travelled = acalc.t_distance([0, 0], target_line[1])
            final_target_ln = target_line[:2]

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

    def update_all_metrics(self, use_filtered=True, redo_target_line=False):
        """
        Updates all metric values on the object.
        """
        if redo_target_line:
            self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
            self.target_rotation = self.generate_target_rot()  # TODO: doesn't work for true cw and ccw yet

        translation_fd, rotation_fd = am.calc_frechet_distance(self)
        # fd = am.calc_frechet_distance_all(self)

        try:
            mvt_efficiency, arc_len = am.calc_mvt_efficiency(self)
        except RuntimeWarning:
            mvt_efficiency = -1
            arc_len = -1

        try:
            max_error = am.calc_max_error(self, arc_len)
        except RuntimeWarning:
            max_error = 0

        try:  # TODO: move all these try excepts to asterisk metrics
            area_btwn = am.calc_area_btwn_curves(self)
        except:
            area_btwn = -1

        try:  # this one is particularly troublesome
            max_area_region, max_area_loc = am.calc_max_area_region(self)
        except IndexError:
            print("Max area region failed")
            max_area_region = -1
            max_area_loc = -1

        # TODO: Make getters for each metric - can also return none if its not calculated
        metric_dict = {"trial": self.generate_name(), "dist": self.total_distance,
                       "t_fd": translation_fd, "r_fd": rotation_fd,  # "fd": fd
                       "max_err": max_error, "mvt_eff": mvt_efficiency, "arc_len": arc_len,
                       "area_btwn": area_btwn, "max_a_reg": max_area_region, "max_a_loc": max_area_loc}

        self.metrics = pd.Series(metric_dict)
        return self.metrics


if __name__ == '__main__':
    test = AstTrial("sub1_2v2_c_n_1.csv", do_metrics=True, norm_data=True)
    #print(test.metrics)
    test.plot_trial(use_filtered=False)
