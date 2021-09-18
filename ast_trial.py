#!/usr/bin/env python3

"""
Handles a single asterisk trial. Manages reading in, conditioning, and filtering object paths. Also handles plotting.
Included are a base class which implements simple
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
from ast_basic import AstBasicData


class AstTrial(AstBasicData):
    """
    Class to represent a single Asterisk Test Trial.
    """
    def __init__(self, file_name, data=None, folder=None, do_metrics=True, norm_data=True,
                 controller_label=None, condition_data=True):
        """ # TODO: redo comment for constructor
        Class to represent a single asterisk test trial.
        """

        self.path_labels = []
        self.poses = None
        self.metrics = None

        self.controller_label = controller_label  # TODO: integrate controller label into the plot title

        self.normalized = norm_data
        self.filtered = False
        self.window_size = 0

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
                self.demographics_from_filename(file_name=file_name)

                self.add_data_by_file(file_name, condition_data=condition_data,
                                      do_metrics=do_metrics, norm_data=norm_data)

            # if a filename is provided and a dataframe is provided
            elif file_name is not None and data is not None and isinstance(file_name, str):
                self.demographics_from_filename(file_name=file_name)

            else:
                raise TypeError("Filename failed.")

            print(self.generate_name())

        except Exception as e:
            print(e)
            print("AstTrial loading failed.")
            raise ImportError("Filename failed, AstTrial failed to generate.")

        # if file_name:
        #     print(self.generate_name())

    def demographics_from_filename(self, file_name):
        """
        Takes a file_name and gets relevant information out
        """
        s, h, t, r, e = file_name.split("_")
        n, _ = e.split(".")

        self.add_hand_info(h)
        self.data_demographics(s, t, r, n)

    def add_data_by_file(self, file_name, norm_data=True, handinfo_name=None, do_metrics=True, condition_data=True):
        """
        Add object path data as a file. By default, will run data through conditioning function
        """
        # Data will not be filtered in this step
        path_df = self._read_file(file_name, condition_data=condition_data, norm_data=norm_data)

        self.poses = path_df[["x", "y", "rmag"]]

        self.target_line, self.total_distance = self.generate_target_line(100, norm_data)  # 100 samples
        self.target_rotation = self.generate_target_rot()

        self.assess_path_labels()
        print(self.path_labels)

        if do_metrics and self.poses is not None and "no_mvt" not in self.path_labels:
            self.update_all_metrics()

    def add_data_by_df(self, path_df, condition_df=True, do_metrics=True, norm_data=True):
        """
        Add object path data as a dataframe. By default, will run dataframe through conditioning function
        """
        path_df = path_df.set_index("frame")

        if condition_df:
            data = self._condition_df(path_df, norm_data)
        else:
            data = path_df

        self.poses = data[["x", "y", "rmag"]]

        self.target_line, self.total_distance = self.generate_target_line(100, norm_data)  # 100 samples
        self.target_rotation = self.generate_target_rot()

        self.assess_path_labels()

        if do_metrics and self.poses is not None and "no_mvt" not in self.path_labels:
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
            df = df.set_index("frame")
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
        #df = df.set_index("frame")

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

    def is_avg_trial(self):
        return False

    def is_standing_rot_trial(self):
        return False

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
        self._plot_orientations(scale=1.0)

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

        plt.title(f"Plot: {self.generate_plot_title()}")

        if provide_notes:
            self._plot_notes()

        if save_plot:
            plt.savefig(f"results/pics/plot_{self.generate_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

if __name__ == '__main__':
    test = AstTrial(file_name="sub1_basic_c_n_2.csv", do_metrics=True, norm_data=False)
    print(f"name: {test.generate_name()}")
    print(f"tot dist: {test.total_distance}")
    print(f"path labels: {test.path_labels}")
    print(f"metrics: {test.metrics}")

    test.moving_average(window_size=10)
    test.plot_trial(use_filtered=False, provide_notes=True)
