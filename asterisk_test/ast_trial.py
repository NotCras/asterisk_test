import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_plotting import AsteriskPlotting as aplt
from data_calculations import AsteriskCalculations as acalc
from metric_calculation import AstMetrics as am
from trial_labelling import AsteriskLabelling as al
import pdb
from ast_hand_info import HandInfo


class AstTrial:
    """
    Base class for Asterisk Trial classes -> AstTrial and AveragedTrial so far
    """
    def __init__(self, file_obj, data=None, subject_label=None,  translation_label=None, rotation_label=None,
                 number_label=None, controller_label=None):
        self.subject, self.trial_translation, self.trial_rotation, \
            self.trial_num, self.controller_label = None, None, None, None, None
        self.data_demographics(subject=subject_label, translation=translation_label,
                               rotation=rotation_label, number=number_label, controller=controller_label)

        self.file_locs = file_obj

        if data is not None:
            self.poses = data[["x", "y", "rmag"]]
        else:
            self.poses=None

        self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
        self.target_rotation = self.generate_target_rot()

        self.path_labels = []
        self.metrics = None

        self.normalized = True
        self.filtered = False
        self.window_size = 0

    def data_demographics(self, subject=None, translation=None, rotation=None, number=None, controller=None):
        """
        Add demographics to your data
        """
        self.subject = subject
        self.trial_translation = translation
        self.trial_rotation = rotation
        self.trial_num = number
        self.controller_label = controller

    def add_data_by_df(self, df):
        self.poses = df[["x", "y", "rmag"]]

        self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
        self.target_rotation = self.generate_target_rot()

    def add_hand_info(self, hand_name):
        """
        If you didn't make the object with a file_name, a function to set hand in painless manner
        :param hand_name: name of hand to make
        """
        self.hand = HandInfo(hand_name)

    def generate_name(self):
        """
        Generates the codified name of the trial
        :return: string name of trial
        """
        return f"{self.hand.get_name()}_{self.trial_translation}_{self.trial_rotation}_" \
               f"{self.subject}_{self.trial_num}"

    def generate_plot_title(self):
        """
        Generates the codified name of the trial for the plot title (it includes controller_label)
        :return: string name of trial
        """
        return f"{self.hand.get_name()}_{self.controller_label}_{self.trial_translation}_" \
               f"{self.trial_rotation}_{self.subject}_{self.trial_num}"

    def is_ast_trial(self):
        return False

    def is_avg_trial(self):
        return False

    def is_standing_rot_trial(self):
        return False

    def get_last_pose(self):
        """
        Returns last pose as an array. Returns both filtered and unfiltered data if obj is filtered
        """
        return self.poses.dropna().tail(1).to_numpy()[0]

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

    def crop_data(self, start_i=1, end_i=None):
        """
        Enables you to crop data
        """
        data_size = self.poses.shape[0]

        if start_i <= 0:
            start_i = 1

        if end_i is None:
            end_i = data_size - 1
        elif end_i >= data_size:
            print(f"Too large ending index")
            end_i = data_size - 1

        self.poses = self.poses.loc[start_i:end_i]

    def generate_target_line(self, n_samples=100, normalized=True):
        """
        Using object trajectory (self.poses), build a line to compare to for frechet distance.
        Updates this attribute on object.
        :param n_samples: number of samples for target line. Defaults to 100
        """

        x_vals, y_vals = aplt.get_direction(self.trial_translation, n_samples=n_samples)

        if not normalized:
            x_vals = x_vals*100  # units are in millimeters, so we end up with the potential for 10 cm of movement
            y_vals = y_vals*100  # may need to keep an eye on this one fyi TODO

        target_line = np.column_stack((x_vals, y_vals))

        # get last object pose and use it for determining how far target line should go
        # last_obj_pose = self.poses.tail(1).to_numpy()[0]
        last_obj_pose = self.get_last_pose()

        target_line_length = acalc.narrow_target(last_obj_pose, target_line)

        if target_line_length <= n_samples - 2 and target_line_length > 0:  # want the plus 1 to round up
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
            target_val = float(self.trial_rotation[1:])
            target_rot = np.array([target_val])

        # elif self.trial_rotation == "m15":
        #     target_rot = np.array([-15])

        else:
            target_rot = np.zeros(1)

        return target_rot

    def assess_path_labels(self, no_mvt_threshold=0.1, init_threshold=0.05, init_num_pts=10, dev_perc_threshold=0.10):
        """
        Assesses the labels on the data, adds labels that fit to path_labels.
        """
        # check whether total distance is an acceptable distance to consider it actually movement
        if self.total_distance < no_mvt_threshold:  # TODO: should this be arc len based? Or incorporate arc len too?
            self.path_labels.append("no_mvt")
            #print(f"No movement detected in {self.generate_name()}. Skipping metric calculation.")

        # check that data starts near center
        if not al.assess_initial_position(self, threshold=init_threshold, to_check=init_num_pts):
            self.path_labels.append("not centered")  # TODO: need to do this on non-normalized, non-relative intial values
            #print(f"Data for {self.generate_name()} failed, did not start at center.")

        deviated, too_deviated, rot_deviated = al.assess_path_deviation(self, rotation_threshold=75)

        if deviated:
            self.path_labels.append("deviated")

        if too_deviated:
            self.path_labels.append("end deviated")

        if rot_deviated:
            self.path_labels.append("rot deviated")

        mvt_observations = al.assess_path_movement(self)  # TODO: make more in depth?

        if "backtracking" in mvt_observations:
            self.path_labels.append("backtracking")

        # if "shuttling" in mvt_observations:
        #     self.path_labels.append("shuttling")

        return self.path_labels

    def check_labels(self, labels_2_check):
        """
        Check whether an asterisk trial object has specific labels. Will tell you which labels were triggered
        """
        result = False
        triggered_labels = []

        if self.path_labels:
            for label in self.path_labels:  # TODO: there's definitely a more elegant way to do this
                if label in labels_2_check:
                    result = True
                    triggered_labels.append(label)

        else:  # there are no labels on the object
            pass

        return result, triggered_labels

    def save_data(self, folder=None, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        :param file_name_overwrite: optional parameter, will save as generate_name unless a different name is specified
        """ # TODO: add ability to specific your own folder

        if folder is None:
            folder = self.file_locs.path_data

        if file_name_overwrite is None:
            new_file_name = f"{self.generate_name()}.csv"

        else:
            new_file_name = f"{file_name_overwrite}.csv"

        # if data has been filtered, we also want to include that in csv generation,
        # otherwise the filtered columns won't exist
        if self.filtered:
            #filtered_file_name = f"{folder}f{self.window_size}_{new_file_name}"
            filtered_name = f"f{self.window_size}_{new_file_name}.csv"
            filtered_file_name = folder / filtered_name

            self.poses.to_csv(filtered_file_name, index=True, columns=[
                "x", "y", "rmag", "f_x", "f_y", "f_rmag"])
        else:

            self.poses.to_csv(folder / new_file_name, index=True, columns=[
                "x", "y", "rmag"])

        # print(f"CSV File generated with name: {new_file_name}")

    def plot_trial(self):
        pass

    def _plot_notes(self):  # TODO: move to aplt, make it take in a list of labels so HandTranslation can also use it
        """
        Plots the labels and trial ID in the upper left corner of the plot
        """
        note = "Labels:"
        for l in self.path_labels:
            note = f"{note} {l} |"

        ax = plt.gca()
        # plt.text(0.1, 0.2, self.generate_name()) #, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))
        plt.text(-0.1, 1.1, note, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))

    def _plot_orientations(self, marker_scale=25, line_length=0.01, positions=[0.3, 0.55, 0.75], scale=0.25):
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
                # print(t)

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

    def update_all_metrics(self, use_filtered=True, redo_target_line=False):
        """
        Updates all metric values on the object.
        """
        if redo_target_line:
            self.target_line, self.total_distance = self.generate_target_line(100)  # 100 samples
            self.target_rotation = self.generate_target_rot()

        translation_fd, fd = am.calc_frechet_distance(self)
        # fd = am.calc_frechet_distance_all(self)

        mvt_efficiency, arc_len = am.calc_mvt_efficiency(self)

        max_error = am.calc_max_error(self, arc_len)
        max_error_rot = am.calc_rot_max_error(self, arc_len)[0]

        area_btwn = am.calc_area_btwn_curves(self)

        # this one is particularly troublesome
        max_area_region, max_area_loc = am.calc_max_area_region(self)

        # TODO: Make getters for each metric - can also return none if its not calculated
        metric_dict = {"trial": self.generate_name(), "dist": self.total_distance,
                       "t_fd": translation_fd, "fd": fd, #"r_fd": rotation_fd,
                       "max_err": max_error, "max_err_rot": max_error_rot, "mvt_eff": mvt_efficiency,
                       "arc_len": arc_len, "area_btwn": area_btwn,
                       "max_a_reg": max_area_region, "max_a_loc": max_area_loc
                       }

        self.metrics = pd.Series(metric_dict)
        return self.metrics
