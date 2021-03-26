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
    def __init__(self, subjects, hand_name, rotation=None):
        """
        Class to hold all the data pertaining to a specific hand.
        Combines data from all subjects
        :param subjects: subjects to include in hand data object
        :param hand_name: name of hand for this object
        """
        self.hand = HandObj(hand_name)
        self.subjects_containing = subjects
        self.data = self._gather_hand_data(subjects, rotation)
        self.filtered = False
        self.window_size = None
        self.averages = []


    def _gather_hand_data(self, subjects, rotation=None):
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
                data_dictionary[key] = self._make_asterisk_trials(subjects, t, r, [1, 2, 3])

        else:  # TODO: also add a check for just cw and ccw
            for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                key = f"{t}_{rotation}"
                data_dictionary[key] = self._make_asterisk_trials(subjects, t, rotation, [1, 2, 3])

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

    def _average_dir(self, translation, rotation, subject=None):
        """
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param translation: trial direction to average
        :param rotation: trial rotation to average
        :param subject: subject or list of subjects to average, optional. If not provided, defaults to all subjects
        :return returns averaged path
        """
        if subject is None:  # get batches of data by trial type, if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(translation, self.subjects_containing, rotation)

        else:
            trials = self._get_ast_dir(translation, subject, rotation)

        average = AveragedTrial()
        # average.make_average_line(trials)
        average.calculate_avg_line(trials)
        return average

    def replace_trial_data(self, trial_obj):
        """
        Delete trial data obj from stored data and replace with new trial data obj
        Gets attributes of obj to delete from the obj passed in
        """
        # TODO: implement this
        pass

    def calc_avg_ast(self, subjects=None, rotation=None):
        """
        calculate and store all averages
        :param subjects: subject(s) to include in the average. Defaults to all subjects in object
        :param rotation: refers to the rotation type ("n", "m15", "p15"). Defaults to all
        """
        averages = []
        if subjects is None:  # if no subjects given, defaults to all subjects
            subjects = self.subjects_containing

        if rotation is None:
            # TODO: make this smarter, so that we base the list on what exists on object
            for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
                avg = self._average_dir(t, r, subjects)
                averages.append(avg)
        else:
            for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                avg = self._average_dir(t, rotation, subjects)
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
        # TODO: plot orientation error
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive",
                  "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

        # plot data
        for i, t in enumerate(trials):
            data_x, data_y, theta = t.get_poses(use_filtered)

            plt.plot(data_x, data_y, color=colors[i], label='trajectory', linestyle=linestyle)

            if stds: # only for AsteriskAverage objs
                t.plot_sd(colors[i])

        # plot target lines as dotted lines
        self.plot_all_target_lines(colors)
        plt.title(f"{self.hand.get_name()} avg asterisk, rot: {trials[0].trial_rotation}")
        plt.xticks(np.linspace(-0.6, 0.6, 13), rotation=30)
        plt.yticks(np.linspace(-0.6, 0.6, 13))
        plt.gca().set_aspect('equal', adjustable='box')
        return plt

    def _make_fd_plot(self, trials):
        """
        make a bar plot with fd values for each direction
        :param trials: a set of asterisktrials or average objects, best if its an asterisk of data with no repeats
        """
        for i, t in enumerate(trials):  # TODO: make it work with plotting multiple values for one label?
            # plot the fd values for that direction
            trial_label = f"{t.trial_translation}_{t.trial_rotation}"
            t_fd = t.translation_fd
            r_fd = t.rotation_fd  # TODO: we don't do anything with r_fd but fd will only return max error anyway

            plt.bar(trial_label, t_fd)

        return plt

    def plot_data_subset(self, subjects, trial_number="1", show_plot=True, save_plot=False):
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
            plt.savefig(f"pics/fullplot4_{self.hand.get_name()}_{subjects}_{trial_number}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

            # TODO: add ability to make comparison plot between n, m15, and p15
            # TODO: have an ability to plot a single average trial

    def plot_avg_data(self, rotation="n", subjects=None, show_plot=True, save_plot=False, linestyle="solid", plot_contributions=True):
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
            avgs = self.calc_avg_ast(subjects, rotation)

        plt = self._make_plot(avgs, use_filtered=False, stds=True, linestyle=linestyle)

        if plot_contributions:
            for a in avgs:
                a.plot_line_contributions()

        plt.title(f"Avg {self.hand.get_name()}, {subjects}, {rotation}")

        if save_plot:
            plt.savefig(f"pics/avgd_{self.hand.get_name()}_{subjects}_{rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def plot_orientation_errors(self, translation, subject=None, rotation="n", show_plot=True, save_plot=False):
        """
        line plot of orientation error throughout a trial for a specific direction
        :param translation: the type of translation
        :param subject: list of subjects. If none is provided, uses all of them
        :param rotation: type of rotation. Defaults to "n"
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        if subject:
            trials = self._get_ast_dir(translation, subject, rotation)
        else:
            trials = self._get_ast_dir(translation, self.subjects_containing, rotation)

        # if self.averages and incl_avg:  # TODO: have an option to include the average?
        #     for a in self.averages:
        #         if a.trial_translation==direction_label and a.trial_rotation==rotation_label:
        #             trials.append(a)

        for t in trials:
            rot_err = t.calc_rot_err()
            # currently using the get_c function to generate a normalized set of x values to use as x values
            x, _ = aplt.get_c(len(rot_err))  # will need to multiply by 2 to get it to go to 1.0 instead of 0.5
            plt.plot(2*x, rot_err, label=f"Orientation Err {t.subject}, trial {t.trial_num}")

        if save_plot:
            if subject:
                plt.savefig(f"pics/angerror_{self.hand.get_name()}_{subject}_{translation}_{rotation}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/angerror_{self.hand.get_name()}_all_{translation}_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_fd_set(self, subjects, trial_number="1", rotation="n", show_plot=True, save_plot=False):
        """  # TODO: still need to test
        plots the frechet distance values of an asterisk of data specified in the parameters
        :param subject: list of subjects. If none is provided, uses all of them
        :param trial_number: the trial number to choose. Defaults to "1"
        :param rotation: type of rotation. Defaults to "n"
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        trials = self._get_ast_set(subjects, trial_number, rotation)
        # dirs = ["a", "b", "c", "d", "e", "f", "g", "h"]  # TODO: add cw and ccw later, once index issue is fixed

        plt = self._make_fd_plot(trials)
        if subjects:
            plt.title(f"FD {self.hand.get_name()}, {subjects}, {rotation}")
        else:
            plt.title(f"Frechet Distance {self.hand.get_name()}, {rotation}")

        if save_plot:
            if subjects:
                plt.savefig(f"pics/fds_{self.hand.get_name()}_{subjects}_{rotation}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/fds_{self.hand.get_name()}_all_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_avg_fd(self, subjects=None, rotation="n", show_plot=True, save_plot=False):
        """  # TODO: still need to test
        plots averaged fd values in a bar chart
        :param subject: list of subjects. If none is provided, uses all of them
        :param rotation: type of rotation. Defaults to "n"
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        trials = self.averages
        # dirs = ["a", "b", "c", "d", "e", "f", "g", "h"]  # TODO: add cw and ccw later, once index issue is fixed

        plt = self._make_fd_plot(trials)
        if subjects:
            plt.title(f"Avg FD {self.hand.get_name()}, {subjects}, {rotation}")
        else:
            plt.title(f"Avg FD {self.hand.get_name()}, {rotation}")

        if save_plot:
            if subjects:
                plt.savefig(f"pics/fd_avg_{self.hand.get_name()}_{subjects}_{rotation}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/fd_avg_{self.hand.get_name()}_all_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_all_target_lines(self, order_of_colors):
        """
        Plot all target lines on a plot for easy reference
        :param order_of_colors:
        """
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
    h = AsteriskHandData(["sub1", "sub2"], "2v3", rotation="n")
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
    h.plot_avg_data(rotation="n", subjects=None, show_plot=True, save_plot=False)


