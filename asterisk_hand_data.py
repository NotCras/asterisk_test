#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math as m
from pathlib import Path
import pdb
import matplotlib.pyplot as plt
import asterisk_data_manager as datamanager
import asterisk_trial as trial
from asterisk_hand import HandObj
from asterisk_average import AveragedTrial
from asterisk_plotting import AsteriskPlotting as aplt


class AsteriskHandData:
    # TODO: add ability to add trials after the fact?
    def __init__(self, subjects, hand_name):
        """
        Class to hold all the data pertaining to a specific hand.
        Combines data from all subjects
        :param subjects: subjects to include in hand data object
        :param hand_name: name of hand for this object
        """
        self.hand = HandObj(hand_name)
        self.subjects_containing = subjects
        self.data = self._gather_hand_data(subjects)
        self.filtered = False
        self.window_size = None
        self.averages = []

    def _gather_hand_data(self, subjects):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        :param subjects: list of subjects to get
        """
        data_dictionary = dict()
        for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
            key = f"{t}_{r}"
            data_dictionary[key] = self._make_asterisk_trials(subjects, t, r, [1, 2, 3])

        return data_dictionary

    def _make_asterisk_trials(self, subjects, translation_label, rotation_label, trials):
        """
        Goes through data and compiles data with set attributes into an AsteriskTrial objects
        :param subjects: name of subject
        :param translation_label: name of translation trials
        :param rotation_label: name of rotation trials
        :param trial_num: trial numbers to include, default parameter
        """
        gathered_data = list()
        for s in subjects:  # TODO: subjects is a list, make a type recommendation?
            for n in trials:
                try:
                    asterisk_trial_file = f"{s}_{self.hand.get_name()}_{translation_label}_{rotation_label}_{n}.csv"

                    trial_data = trial.AsteriskTrialData(asterisk_trial_file)

                    gathered_data.append(trial_data)

                except Exception as e:
                    print(e)
                    print("Skipping.")
                    continue

        return gathered_data

    def add_trial(self, ast_trial):
        """
        add an ast_trial after the asteriskhanddata object was created
        :param ast_trial: asterisktrialdata to add
        """
        label = f"{ast_trial.trial_translation}_{ast_trial.trial_rotation}"
        self.data[label].append(ast_trial)

    def _get_ast_batch(self, subjects, trial_number=None, rotation_type="n"):  # TODO: rename this function, be more specific
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
                        break
                    # TODO: throw an exception in case there isn't any of the trial that we want

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

    def _average_dir(self, translation, rotation, subject=None):
        """  # TODO: still need to test
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param translation: trial direction to average
        :param rotation: trial rotation to average
        :param subject: subject or list of subjects to average, optional. If not provided, defaults to all subjects
        :return returns averaged path
        """
        if subject:  # get batches of data by trial type
            trials = self._get_ast_dir(translation, subject, rotation)
        else:  # if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(translation, self.subjects_containing, rotation)

        average = AveragedTrial()
        average.make_average_line(trials)
        return average

    def calc_avg_ast(self, subjects=None, rotation=None):
        """
        calculate and store all averages
        :param subjects: subject(s) to include in the average. Defaults to all subjects in object
        :param rotation: refers to the rotation type ("n", "m15", "p15"). Defaults to all
        """
        dfs = []
        if subjects:
            pass
        else:  # if no subjects given, defaults to all subjects
            subjects = self.subjects_containing

        if rotation:
            for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                avg = self._average_dir(t, rotation, subjects)
                dfs.append(avg)
        else:
            for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
                avg = self._average_dir(t, r, subjects)
                dfs.append(avg)

        self.averages = dfs
        return dfs

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

    def _make_plot(self, trials, use_filtered=True, stds=False):
        """
        Function to make our plots.
        :param trials: either a list of AsteriskTrialData or AsteriskAverage objs
        :param use_filtered: flag whether to use filtered data. Default is True
        :param stds: flag whether to plot standard deviations. Only for AsteriskAverage objects. Default is False
        """
        # TODO: plot orientation error
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive",
                  "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

        # plot data
        for i, t in enumerate(trials):
            data_x, data_y, theta = t.get_poses(use_filtered)
            plt.plot(data_x, data_y, color=colors[i], label='trajectory')

            if stds:
                t.plot_sd(colors[i])

        # plot target lines as dotted lines
        self.plot_all_target_lines(colors)
        plt.xticks(np.linspace(-0.6, 0.6, 11), rotation=30)
        plt.yticks(np.linspace(-0.6, 0.6, 11))
        return plt

    def _make_fd_plot(self, trials):
        """
        make a bar plot with fd values for each direction
        :param trials: a set of asterisktrials or average objects, best if its an asterisk of data with no repeats
        """
        for i, t in enumerate(trials):
            # plot the fd values for that direction
            trial_label = f"{t.trial_translation}_{t.trial_rotation}"
            t_fd = t.translation_fd
            r_fd = t.rotation_fd  # TODO: don't do anything with r_fd

            plt.bar(i, t_fd, trial_label)

        return plt

    def plot_data_subset(self, subjects, trial_number="1", show_plot=True, save_plot=False):
        """
        Plots a subset of the data, as specified in parameters
        :param subjects: subjects or list of subjects,
        :param trial_number: the number of trial to include
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        dfs = self._get_ast_batch([subjects], trial_number)  # TODO: make this work for the hand data object
        plt = self._make_plot(dfs)
        plt.title(f"Plot: {subjects}_{self.hand.get_name()}, set #{trial_number}")

        if save_plot:
            plt.savefig(f"pics/fullplot4_{subjects}_{self.hand.get_name()}_{trial_number}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_avg_data(self, subjects, rotation="n", show_plot=True, save_plot=False):
        """
        Plots the data from one subject, averaging all of the data in each direction
        :param subjects list of subjects. If none is provided, uses all of them
        :param rotation the type of rotation type to plot, will collect an asterisk of this
        :param show_plot flag to show plot. Default is true
        :param save_plot flat to save plot as a file. Default is False
        """
        avgs = self.calc_avg_ast(subjects, rotation)
        plt = self._make_plot(avgs, use_filtered=False, stds=True)

        if subjects:
            plt.title(f"{subjects}, {self.hand.get_name()}, {rotation}")
        else:
            plt.title(f"Averaged {self.hand.get_name()}, {rotation}")

        if save_plot:
            if subjects:
                plt.savefig(f"pics/avgd_{subjects}_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/avgd_all_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_orientation_error(self, translation, subject=None, rotation="n", show_plot=True, save_plot=False):
        """  # TODO: still need to test
        line plot of orientation error throughout a trial for a specific direction
        :param translation the type of translation
        :param subject list of subjects. If none is provided, uses all of them
        :param rotation type of rotation. Defaults to "n"
        :param show_plot flag to show plot. Default is true
        :param save_plot flat to save plot as a file. Default is False
        """
        trials = self._get_ast_dir(translation, subject, rotation)

        # if self.averages and incl_avg:  # TODO: have an option to include the average?
        #     for a in self.averages:
        #         if a.trial_translation==direction_label and a.trial_rotation==rotation_label:
        #             trials.append(a)

        for t in trials:
            rot_err = t.calc_rot_err()  # TODO: not sure what to use as an x value for orientation error
            x, _, _ = t.get_poses()
            plt.plt(x, rot_err, label=f"{t.subject}, trial {t.trial_num}")

        if save_plot:
            if subject:
                plt.savefig(f"pics/angerror_{subject}_{translation}_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/angerror_all_{translation}_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_fd_set(self, subjects, trial_number="1", rotation="n", show_plot=True, save_plot=False):
        """  # TODO: still need to test
        plots the frechet distance values of an asterisk of data specified in the parameters
        :param subject list of subjects. If none is provided, uses all of them
        :param trial_number the trial number to choose. Defaults to "1"
        :param rotation type of rotation. Defaults to "n"
        :param show_plot flag to show plot. Default is true
        :param save_plot flat to save plot as a file. Default is False
        """
        trials = self._get_ast_batch(subjects, trial_number, rotation)
        # dirs = ["a", "b", "c", "d", "e", "f", "g", "h"]  # TODO: add cw and ccw later, once index issue is fixed

        plt = self._make_fd_plot(trials)
        if subjects:
            plt.title(f"FD {subjects}, {self.hand.get_name()}, {rotation}")
        else:
            plt.title(f"Frechet Distance {self.hand.get_name()}, {rotation}")

        if save_plot:
            if subjects:
                plt.savefig(f"pics/fds_{subjects}_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/fds_all_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_avg_fd(self, subjects=None, rotation="n", show_plot=True, save_plot=False):
        """  # TODO: still need to test
        plots averaged fd values in a bar chart
        :param subject list of subjects. If none is provided, uses all of them
        :param rotation type of rotation. Defaults to "n"
        :param show_plot flag to show plot. Default is true
        :param save_plot flat to save plot as a file. Default is False
        """
        trials = self.averages
        # dirs = ["a", "b", "c", "d", "e", "f", "g", "h"]  # TODO: add cw and ccw later, once index issue is fixed

        plt = self._make_fd_plot(trials)
        if subjects:
            plt.title(f"Avg FD {subjects}, {self.hand.get_name()}, {rotation}")
        else:
            plt.title(f"Avg FD {self.hand.get_name()}, {rotation}")

        if save_plot:
            if subjects:
                plt.savefig(f"pics/fd_avg_{subjects}_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/fd_avg_all_{rotation}_{self.hand.get_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_all_target_lines(self, order_of_colors):
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


if __name__ == '__main__':
    h = AsteriskHandData(["sub1", "sub2"], "2v2")
    h.filter_data()

    # subject 1 averages
    h.plot_avg_data(subjects="sub1", rotation="n", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects="sub1", rotation="m15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects="sub1", rotation="p15", show_plot=False, save_plot=True)
    plt.clf()

    # subject 2 averages
    h.plot_avg_data(subjects="sub2", rotation="n", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects="sub2", rotation="m15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects="sub2", rotation="p15", show_plot=False, save_plot=True)
    plt.clf()

    # all subjects
    h.plot_avg_data(subjects=None, rotation="p15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects=None, rotation="m15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects=None, rotation="n", show_plot=True, save_plot=True)


