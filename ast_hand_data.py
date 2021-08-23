#!/usr/bin/env python3
"""
Class for organizing asterisk trial data for one specific hand. Handles analysis, averaging, and plotting.
"""

import numpy as np
import pandas as pd
import math as m
from pathlib import Path
import csv
import pdb
import matplotlib.pyplot as plt
import data_manager as datamanager
import ast_trial as trial
from ast_hand_info import HandInfo
from ast_averaging import AveragedTrial
from data_plotting import AsteriskPlotting as aplt


class AstHandTrials:
    # TODO: add ability to add trials after the fact?
    def __init__(self, subjects, hand_name, rotation=None, blocklist_file=None):
        """
        Class to hold all the data pertaining to a specific hand.
        Combines data from all subjects
        :param subjects: subjects to include in hand data object
        :param hand_name: name of hand for this object
        """
        self.hand = HandInfo(hand_name)
        self.subjects_containing = subjects
        if blocklist_file is not None:
            blocklist = self._check_blocklist(blocklist_file)
        else:
            blocklist = None

        self.data = self._gather_hand_data(subjects, rotation, blocklist=blocklist)
        self.filtered = False
        self.window_size = None
        self.averages = []

    def _check_blocklist(self, file_name):
        """
        Checks blocklist file to get the list of trials that should not be included.
        """
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            blocked_files = []
            for row in csv_reader:
                blocked_files.append(row[0])

            print(f"Will block these files: {blocked_files}")

        return blocked_files

    def _gather_hand_data(self, subjects, rotation=None, blocklist=None):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        :param subjects: list of subjects to get
        """
        data_dictionary = dict()
        if rotation is None:
            for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
                key = f"{t}_{r}"
                data = self._make_asterisk_trials(subjects, t, r,
                                                                  datamanager.generate_options("numbers"),
                                                                  blocklist=blocklist)
                if data:
                    data_dictionary[key] = data
                else:
                    print(f"{key} not included, no valid data")

        elif rotation in datamanager.generate_options("rotations"):
            for t in datamanager.generate_options("translations"):
                key = f"{t}_{rotation}"
                data = self._make_asterisk_trials(subjects, t, rotation,
                                                                  datamanager.generate_options("numbers"),
                                                                  blocklist=blocklist)
                if data:
                    data_dictionary[key] = data
                    # pdb.set_trace()
                else:
                    print(f"{key} not included, no valid data")
                    # pdb.set_trace()

        elif rotation in datamanager.generate_options("rotations_n_trans"):
            key = f"n_{rotation}"
            data = self._make_asterisk_trials(subjects, "n", rotation,
                                                              datamanager.generate_options("numbers"),
                                                              blocklist=blocklist)
            if data:
                data_dictionary[key] = data
            else:
                print(f"{key} not included, no valid data")

        else:
            print("invalid key")
            data_dictionary = None

        return data_dictionary

    def _get_directions_in_data(self):
        """
        Returns a list of trial directions that exist in the data
        :return:
        """
        list_of_dirs = list()
        for k in list(self.data.keys()):
            if k[0] != "n":
                list_of_dirs.append(k[0])

        return list_of_dirs

    def _make_asterisk_trials(self, subjects, translation_label, rotation_label, trials, blocklist=None):
        """
        Goes through data and compiles data with set attributes into an AsteriskTrial objects
        :param subjects: name of subject
        :param translation_label: name of translation trials
        :param rotation_label: name of rotation trials
        :param trial_num: trial numbers to include, default parameter
        """
        # TODO: Change things so that we have some way to know when the trials were all no mvt or deviation
        # TODO: make a report txt file which sa
        # Maybe make it return None? then we can return all dictionary keys that don't return none in the other func

        gathered_data = list()
        for s in subjects:  # TODO: subjects is a list, make a type recommendation?
            for n in trials:
                asterisk_trial = f"{s}_{self.hand.get_name()}_{translation_label}_{rotation_label}_{n}"

                if blocklist is not None and asterisk_trial in blocklist:
                    print(f"{asterisk_trial} is blocklisted and will not be used.")
                    continue

                try:
                    trial_data = trial.AstTrial(f"{asterisk_trial}.csv")
                    print(f"{trial_data.generate_name()}, labels: {trial_data.path_labels}")

                    # TODO: remove this exclusion here, add into averaging and plotting functions an exclude parameter
                    # which will take a list of path_labels that we should not include in the formulation
                    # have an attribute which shows which labels are excluded from the averaging so that
                    # we know when to re-rerun averaging
                    if "no_mvt" not in trial_data.path_labels and "deviation" not in trial_data.path_labels:

                        gathered_data.append(trial_data)

                    else:
                        print(f"{trial_data.generate_name()} failed (no mvt or deviation), not including file.")
                        continue
                    #print(" ")

                except Exception as e:
                    print(e)
                    print("Skipping.")
                    #print(" ")
                    continue

        return gathered_data

    def add_trial(self, ast_trial):
        """
        add an ast_trial after the asteriskhanddata object was created
        :param ast_trial: asterisktrialdata to add
        """
        label = f"{ast_trial.trial_translation}_{ast_trial.trial_rotation}"
        self.data[label].append(ast_trial)

    def _get_ast_set(self, subjects, trial_number=None, rotation_type="n"):
        """
        Picks out an asterisk of data (all translational directions) with specific parameters
        :param subjects: specify the subject or subjects you want
        :param trial_number: specify the number trial you want, if None then it will
            return all trials for a specific subject
        :param rotation_type: rotation type of batch. Defaults to "n"
        """
        dfs = []
        translations = ["a", "b", "c", "d", "e", "f", "g", "h"]

        for direction in translations:
            dict_key = f"{direction}_{rotation_type}"
            trials = self.data[dict_key]
            # print(f"For {subject_to_run} and {trial_number}: {direction}")

            for t in trials:
                # print(t.generate_name())
                if trial_number:  # if we want a specific trial, look for it
                    if (t.subject == subjects) and (t.trial_num == trial_number):
                        dfs.append(t)
                    elif (t.subject in subjects) and (t.trial_num == trial_number):
                        dfs.append(t)

                else:  # otherwise, grab trial as long as it has the right subject
                    if t.subject == subjects or t.subject in subjects:
                        dfs.append(t)

        return dfs

    def _get_ast_dir(self, direction_label, subjects, rotation_label="n"):
        """
        Get all of the trials for a specific direction. You can specify subject too
        :param direction_label: translation direction
        :param subjects: subject or list of subjects to include
        :param rotation_label: rotation label, defaults to "n"
        """
        dict_key = f"{direction_label}_{rotation_label}"
        direction_trials = self.data[dict_key]
        dfs = []

        for t in direction_trials:
            if t.subject == subjects or t.subject in subjects:
                dfs.append(t)

        return dfs

    def replace_trial_data(self, trial_obj):
        """
        Delete trial data obj from stored data and replace with new trial data obj
        Gets attributes of obj to delete from the obj passed in
        """
        # TODO: implement this
        pass

    def _average_dir(self, translation, rotation, subject=None):
        """
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param translation: trial direction to average
        :param rotation: trial rotation to average
        :param subject: subject or list of subjects to average, optional. If not provided, defaults to all subjects
        :return returns averaged path
        """
        if subject is None:  # get batches of data by trial type, if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(direction_label=translation, subjects=self.subjects_containing,
                                       rotation_label=rotation)

        else:
            trials = self._get_ast_dir(direction_label=translation, subjects=subject, rotation_label=rotation)

        average = AveragedTrial()
        # average.make_average_line(trials)
        if trials:
            average.calculate_avg_line(trials)
            return average

        else:
            print(f"No trials for {translation}_{rotation}, skipping averaging.")
            return None

    def calc_averages(self, subjects=None, rotation=None):
        """
        calculate and store all averages
        :param subjects: subject(s) to include in the average. Defaults to all subjects in object
        :param rotation: refers to the rotation type ("n", "m15", "p15"). Defaults to all options
        """
        averages = []
        if subjects is None:  # if no subjects given, defaults to all subjects
            subjects = self.subjects_containing

        dirs = self._get_directions_in_data()
        print(f"Directions included: {dirs}")

        if rotation is None:
            # TODO: make this smarter, so that we base the list on what exists on object
            for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
                # make sure that we only include translations that are in the data
                if t in dirs:  # TODO: also look at rotations
                    print(f"Averaging {t}")
                    avg = self._average_dir(t, r, subjects)
                    if avg is not None:
                        averages.append(avg)
        else:
            for t in datamanager.generate_options("translations"):
                # make sure that we only include translations that are in the data
                if t in dirs:
                    print(f"Averaging {t}")
                    avg = self._average_dir(translation=t, rotation=rotation, subject=subjects)
                    if avg is not None:
                        averages.append(avg)

        self.averages = averages
        return averages

    def filter_data(self, window_size=15):
        """
        Runs moving average on data stored inside object
        :param window_size: size of moving average. default is 15
        """
        for key in self.data.keys():
            for t in self.data[key]:
                t.moving_average(window_size)

        self.filtered = True
        self.window_size = window_size

    def save_all_data(self):
        """
        Saves each AsteriskTrialObject as a csv file
        """
        for key in self.data.keys():
            for t in self.data[key]:
                t.save_data()
                # print(f"Saved: {t.generate_name()}")

    def _make_plot(self, trials, use_filtered=True, stds=False, linestyle="solid"):
        """
        Function to make our plots.
        :param trials: either a list of AsteriskTrialData or AsteriskAverage objs
        :param use_filtered: flag whether to use filtered data. Default is True
        :param stds: flag whether to plot standard deviations. Only for AsteriskAverage objects. Default is False
        """
        # TODO: plot orientation error?
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

        plt.figure(figsize=(7, 7))

        # get all the averages that we have
        avg_labels = list()
        for a in self.averages:
            avg_labels.append(a.trial_translation)

        if len(avg_labels) == 8:
            # plot target lines as dotted lines
            self.plot_all_target_lines(colors)  # TODO: maybe make set colors for each direction
        else:
            self.plot_all_target_lines(colors, avg_labels)

        # plot data
        for i, t in enumerate(trials):
            data_x, data_y, theta = t.get_poses(use_filtered)

            plt.plot(data_x, data_y, color=colors[i], label='trajectory', linestyle=linestyle)

            # plot orientation error
            t._plot_orientations(marker_scale=15, line_length=0.025, scale=1)

            if stds:  # only for AsteriskAverage objs
                t.plot_sd(colors[i])

        plt.title(f"{self.hand.get_name()} avg asterisk")  # , rot: {trials[0].trial_rotation}")
        plt.xticks(np.linspace(-0.6, 0.6, 13), rotation=30)
        plt.yticks(np.linspace(-0.6, 0.6, 13))
        plt.gca().set_aspect('equal', adjustable='box')
        return plt

    def plot_ast_subset(self, subjects, trial_number="1", show_plot=True, save_plot=False):
        """
        Plots a subset of the data, as specified in parameters
        :param subjects: subjects or list of subjects,
        :param trial_number: the number of trial to include
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        dfs = self._get_ast_set(subjects, trial_number)  # TODO: make this work for the hand data object
        plt = self._make_plot(dfs)
        plt.title(f"Plot: {self.hand.get_name()}, {subjects}, set #{trial_number}")

        if save_plot:
            plt.savefig(f"results/pics/fullplot4_{self.hand.get_name()}_{subjects}_{trial_number}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

            # TODO: add ability to make comparison plot between n, m15, and p15
            # TODO: have an ability to plot a single average trial

    def plot_ast_avg(self, rotation="n", subjects=None, show_plot=True, save_plot=False,
                     linestyle="solid", plot_contributions=False):
        """
        Plots the data from one subject, averaging all of the data in each direction
        :param subjects: list of subjects. If none is provided, uses all of them
        :param rotation: the type of rotation type to plot, will collect an asterisk of this
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        if subjects is None:
            subjects = self.subjects_containing

        # TODO: check that specifying subjects works ok when an average was already calculated
        if self.averages:
            avgs = self.averages

        else:
            avgs = self.calc_averages(subjects=subjects, rotation=rotation)

        plt = self._make_plot(avgs, use_filtered=False, stds=True, linestyle=linestyle)

        if plot_contributions:
            for a in avgs:
                a._plot_line_contributions()

        # TODO: add orientation markers to each line so we have some idea of orientation along the path
        # TODO: add attributes for object shape, size, and initial position!
        plt.title(f"Avg {self.hand.get_name()}, {subjects}, {rotation}, Cube (0.25 span), 0.75 depth init pos")

        if save_plot:
            plt.savefig(f"results/pics/avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def plot_all_target_lines(self, order_of_colors, specific_lines=None):
        """
        Plot all target lines on a plot for easy reference
        :param order_of_colors:
        """
        if specific_lines is None:
            x_a, y_a = aplt.get_a()
            x_b, y_b = aplt.get_b()
            x_c, y_c = aplt.get_c()
            x_d, y_d = aplt.get_d()
            x_e, y_e = aplt.get_e()
            x_f, y_f = aplt.get_f()
            x_g, y_g = aplt.get_g()
            x_h, y_h = aplt.get_h()

            ideal_xs = [x_a, x_b, x_c, x_d, x_e, x_f, x_g, x_h]
            ideal_ys = [y_a, y_b, y_c, y_d, y_e, y_f, y_g, y_h]

            for i in range(8):
                plt.plot(ideal_xs[i], ideal_ys[i], color=order_of_colors[i], label='ideal', linestyle='--')

        else:  # there are specific directions you want to plot, and only those directions
            ideal_xs = list()
            ideal_ys = list()

            if "a" in specific_lines:
                x_a, y_a = aplt.get_a()
                ideal_xs.append(x_a)
                ideal_ys.append(y_a)

            if "b" in specific_lines:
                x_b, y_b = aplt.get_b()
                ideal_xs.append(x_b)
                ideal_ys.append(y_b)

            if "c" in specific_lines:
                x_c, y_c = aplt.get_c()
                ideal_xs.append(x_c)
                ideal_ys.append(y_c)

            if "d" in specific_lines:
                x_d, y_d = aplt.get_d()
                ideal_xs.append(x_d)
                ideal_ys.append(y_d)

            if "e" in specific_lines:
                x_e, y_e = aplt.get_e()
                ideal_xs.append(x_e)
                ideal_ys.append(y_e)

            if "f" in specific_lines:
                x_f, y_f = aplt.get_f()
                ideal_xs.append(x_f)
                ideal_ys.append(y_f)

            if "g" in specific_lines:
                x_g, y_g = aplt.get_g()
                ideal_xs.append(x_g)
                ideal_ys.append(y_g)

            if "h" in specific_lines:
                x_h, y_h = aplt.get_h()
                ideal_xs.append(x_h)
                ideal_ys.append(y_h)

            for i in range(len(ideal_xs)):
                plt.plot(ideal_xs[i], ideal_ys[i], color=order_of_colors[i], label='ideal', linestyle='--')

if __name__ == '__main__':
    h = AstHandTrials(["sub1", "sub2", "sub3"], "2v2", rotation="n")
    h.filter_data()

    # # subject 1 averages
    # h.plot_avg_data(rotation="n", subjects="sub1", show_plot=False, save_plot=True)
    # plt.clf()
    # h.plot_avg_data(rotation="m15", subjects="sub1", show_plot=False, save_plot=True)
    # plt.clf()
    # h.plot_avg_data(rotation="p15", subjects="sub1", show_plot=False, save_plot=True)
    # plt.clf()
    #
    # # subject 2 averages
    # h.plot_avg_data(rotation="n", subjects="sub2", show_plot=False, save_plot=True)
    # plt.clf()
    # h.plot_avg_data(rotation="m15", subjects="sub2", show_plot=False, save_plot=True)
    # plt.clf()
    # h.plot_avg_data(rotation="p15", subjects="sub2", show_plot=False, save_plot=True)
    # plt.clf()
    #
    # # all subjects
    # h.plot_avg_data(rotation="p15", subjects=None, show_plot=False, save_plot=True)
    # plt.clf()
    # h.plot_avg_data(rotation="m15", subjects=None,  show_plot=False, save_plot=True)
    # plt.clf()
    h.plot_ast_avg(rotation="n", subjects=None, show_plot=True, save_plot=False)


