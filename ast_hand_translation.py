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


class AstHandTranslation:
    def __init__(self, subjects, hand_name, rotation='n', blocklist_file=None):
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

        self.set_rotation = rotation
        self.data = self._gather_hand_data(subjects, blocklist=blocklist)
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

    def _gather_hand_data(self, subjects, blocklist=None):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        :param subjects: list of subjects to get
        """
        data_dictionary = dict()

        for t in datamanager.generate_options("translations"):
            key = f"{t}_{self.set_rotation}"
            data = self._make_asterisk_trials_from_filenames(subjects, t, self.set_rotation,
                                                             datamanager.generate_options("numbers"),
                                                             blocklist=blocklist)
            if data:
                data_dictionary[key] = data
                # pdb.set_trace()
            else:
                print(f"{key} not included, no valid data")
                # pdb.set_trace()

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

    def _make_asterisk_trials_from_filenames(self, subjects, translation_label, rotation_label, trials, blocklist=None):
        """
        Goes through data and compiles data with set attributes into an AsteriskTrial objects
        :param subjects: name of subject
        :param translation_label: name of translation trials
        :param rotation_label: name of rotation trials
        :param trial_num: trial numbers to include, default parameter
        """
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

                    gathered_data.append(trial_data)

                except Exception as e:
                    print(f"AstTrial generation failed for {asterisk_trial}")
                    print(e)
                    #print(" ")
                    continue

        return gathered_data

    def _import_data_from_ast_trial_list(self, trial_list):  # TODO: to implement later
        pass

    def add_trial(self, ast_trial):
        """
        add an ast_trial after the asteriskhanddata object was created
        :param ast_trial: asterisktrialdata to add
        """
        label = f"{ast_trial.trial_translation}_{ast_trial.trial_rotation}"
        self.data[label].append(ast_trial)

    def _get_ast_set(self, subjects, trial_number=None, exclude_path_labels=None):
        """
        Picks out an asterisk of data (all translational directions) with specific parameters
        :param subjects: specify the subject or subjects you want
        :param trial_number: specify the number trial you want, if None then it will
            return all trials for a specific subject
        :param rotation_type: rotation type of batch. Defaults to "n"
        """
        dfs = []
        translations = datamanager.generate_options("translations")  # ["a", "b", "c", "d", "e", "f", "g", "h"]

        for direction in translations:
            dict_key = f"{direction}_{self.set_rotation}"
            # TODO: maybe we set the rotation type per hand trial... might be easier to handle
            trials = self.data[dict_key]
            # print(f"For {subject_to_run} and {trial_number}: {direction}")

            for t in trials:
                # print(t.generate_name())
                if trial_number:  # if we want a specific trial, look for it
                    if (t.subject == subjects) and (t.trial_num == trial_number):
                        for l in t.path_labels:
                            if exclude_path_labels is not None and l in exclude_path_labels:
                                continue  # skip trial if it has that path_label

                        dfs.append(t)
                    elif (t.subject in subjects) and (t.trial_num == trial_number):
                        for l in t.path_labels:
                            if exclude_path_labels is not None and l in exclude_path_labels:
                                continue  # skip trial if it has that path_label

                        dfs.append(t)

                else:  # otherwise, grab trial as long as it has the right subject
                    if t.subject == subjects or t.subject in subjects:
                        for l in t.path_labels:
                            if exclude_path_labels is not None and l in exclude_path_labels:
                                continue  # skip trial if it has that path_label

                        dfs.append(t)

        return dfs

    def _get_ast_dir(self, direction_label, subjects, exclude_path_labels=None):
        """
        Get all of the trials for a specific direction. You can specify subject too
        :param direction_label: translation direction
        :param subjects: subject or list of subjects to include
        :param rotation_label: rotation label, defaults to "n"
        """
        dict_key = f"{direction_label}_{self.set_rotation}"
        direction_trials = self.data[dict_key]
        gotten_trials = []

        for t in direction_trials:
            if t.subject == subjects or t.subject in subjects:
                # check if trial has a path_label that we don't want to include
                for l in t.path_labels:
                    if exclude_path_labels is not None and l in exclude_path_labels:
                        continue  # skip trial if it has that path_label

                # if it passes path_label check, add it to the
                gotten_trials.append(t)

        return gotten_trials

    def replace_trial_data(self, trial_obj):
        """
        Delete trial data obj from stored data and replace with new trial data obj
        Gets attributes of obj to delete from the obj passed in
        """
        # TODO: implement this
        pass

    def _average_dir(self, translation, subject=None, exclude_path_labels=None):
        """
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param translation: trial direction to average
        :param rotation: trial rotation to average
        :param subject: subject or list of subjects to average, optional. If not provided, defaults to all subjects
        :return returns averaged path
        """
        if subject is None:  # get batches of data by trial type, if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(direction_label=translation, subjects=self.subjects_containing,
                                       exclude_path_labels=exclude_path_labels)

        else:
            trials = self._get_ast_dir(direction_label=translation, subjects=subject,
                                       exclude_path_labels=exclude_path_labels)

        if trials:
            average = AveragedTrial(trials=trials)
            return average

        else:
            print(f"No trials for {translation}_{self.set_rotation}, skipping averaging.")
            return None

    def calc_averages(self, subjects=None, exclude_path_labels=None):
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

        for t in datamanager.generate_options("translations"):
            # make sure that we only include translations that are in the data
            if t in dirs:
                print(f"Averaging {t}")
                avg = self._average_dir(translation=t, subject=subjects,
                                        exclude_path_labels=exclude_path_labels)
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
                print(f"Moving Average of size {window_size} is on {t.generate_name()}")
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
        for a in self.averages:  # TODO: what if we have no averages?
            avg_labels.append(a.trial_translation)

        if len(avg_labels) == 8:
            # plot target lines as dotted lines
            self.plot_all_target_lines()
        else:
            self.plot_all_target_lines(specific_lines=avg_labels)

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

    def plot_ast_avg(self, subjects=None, show_plot=True, save_plot=False, include_notes=True,
                     linestyle="solid", plot_contributions=False, exclude_path_labels=None):
        """
        Plots the data from one subject, averaging all of the data in each direction
        :param subjects: list of subjects. If none is provided, uses all of them
        :param rotation: the type of rotation type to plot, will collect an asterisk of this
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """

        # TODO: should we do the same for filtered vs unfiltered?
        if self.averages and subjects is None:
            # if we have averages and the user does not specify subjects just use the averages we have
            subjects = self.subjects_containing
            avgs = self.averages

        elif self.averages and subjects is not None:
            # if we have averages but the user specifies specific subjects, rerun averaage
            avgs = self.calc_averages(subjects=subjects, exclude_path_labels=exclude_path_labels)

        else:
            # otherwise just run the average on everything
            subjects = self.subjects_containing
            avgs = self.calc_averages(subjects=subjects, exclude_path_labels=exclude_path_labels)

        plt = self._make_plot(avgs, use_filtered=False, stds=True, linestyle=linestyle)

        if include_notes:
            self._plot_notes()

        if plot_contributions:
            for a in avgs:
                a._plot_line_contributions()

        # TODO: add orientation markers to each line so we have some idea of orientation along the path
        # TODO: add attributes for object shape, size, and initial position!
        plt.title(f"Avg {self.hand.get_name()}, {subjects}, {self.set_rotation}, Cube (0.25 span), 0.75 depth init pos")

        if save_plot:
            plt.savefig(f"results/pics/avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.set_rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def _plot_notes(self):  # TODO: move to aplt, make it take in a list of labels so HandTranslation can also use it
        """
        Plots the labels and trial ID in the upper left corner of the plot
        """
        note = "Labels:"

        labels = set()
        for a in self.averages:
            for l in a.trialset_labels:
                labels.add(l)

        for l in list(labels):
            note = f"{note} {l} |"

        ax = plt.gca()
        # plt.text(0.1, 0.2, self.generate_name()) #, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))
        plt.text(-0.1, 1.1, note, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))

    def plot_all_target_lines(self, specific_lines=None):
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

            dirs = datamanager.generate_options("translations")
            for i, d in enumerate(dirs):
                plt.plot(ideal_xs[i], ideal_ys[i], color=aplt.get_dir_color(d), label='ideal', linestyle='--')

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

            for i, dir in specific_lines:
                plt.plot(ideal_xs[i], ideal_ys[i], color=aplt.get_dir_color(dir), label='ideal', linestyle='--')

if __name__ == '__main__':
    h = AstHandTranslation(["sub1", "sub2", "sub3"], "2v2", rotation="p15")
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
    h.plot_ast_avg(subjects=None, show_plot=True, save_plot=False)

