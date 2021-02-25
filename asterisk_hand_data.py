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
    # TODO: add ability to add trials after the fact
    def __init__(self, subjects, hand_name):
        """
        Class to hold all the data pertaining to a specific hand.
        Combines data from all subjects
        """
        self.hand = HandObj(hand_name)
        self.subjects_containing = subjects
        self.data = self._gather_hand_data(subjects)
        self.filtered = False
        self.window_size = None
        self.averages = []

    def _gather_hand_data(self, subjects_to_get):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        """
        data_dictionary = dict()
        for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
            key = f"{t}_{r}"
            data_dictionary[key] = self._make_asterisk_trials(subjects_to_get, t, r, [1, 2, 3])

        return data_dictionary

    def _make_asterisk_trials(self, subjects, translation_label, rotation_label, trials):
        """
        Goes through data and compiles data with set attributes into an AsteriskTrial objects
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

    def _get_ast_batch(self, subject_to_run, trial_number=None):  # TODO: rename this function, be more specific
        """
        Picks out the specific subject and trial number from data.
        :param subject_to_run specify the subject you want, in list form! Doesn't matter if only one
        :param trial_number specify the number trial you want, if None then it will return all trials for a specific subject
        """
        dfs = []
        translations = ["a", "b", "c", "d", "e", "f", "g", "h"]

        for dir in translations:
            dict_key = f"{dir}_n"
            trials = self.data[dict_key]
            print(f"For {subject_to_run} and {trial_number}: {dir}")

            for t in trials:
                print(t.generate_name())
                if trial_number:  # if we want a specific trial, look for it
                    if (t.subject == subject_to_run) and (t.trial_num == trial_number):
                        dfs.append(t)
                        print(" ")
                        break
                    # TODO: throw an exception in case there isn't the trial that we want

                else:  # otherwise, grab trial as long as it has the right subject
                    if t.subject == subject_to_run or t.subject in subject_to_run:
                        dfs.append(t)

            # print("    ")

        print(dfs)
        return dfs

    def _get_ast_dir(self, direction_label, subject, rotation_label="n"):
        """
        Get all of the trials for a specific direction. You can specify subject too
        """
        dict_key = f"{direction_label}_{rotation_label}"
        direction_trials = self.data[dict_key]
        dfs = []

        for t in direction_trials:
            if t.subject == subject or t.subject in subject:
                dfs.append(t)

        return dfs

    def _average_dir(self, t, r, subject=None):  # TODO: still need to get this working
        """
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param t, r indicate which you want to
        :return returns averaged path
        """
        # get batches of data by trial type
        if subject:
            trials = self._get_ast_dir(t, subject, r)
        else:  # if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(t, self.subjects_containing, r)

        average = AveragedTrial()  # maybe this goes into a new AsteriskAverage class, just like Cindy
        average.make_average_line(trials)
        return average

    def calc_all_avgs(self, subjects_to_run=None, rot=None):
        """
        calculate and store all averages
        """
        dfs = []

        if subjects_to_run:
            pass
        else:  # if no subjects given, defaults to all subjects
            subjects_to_run = self.subjects_containing

        if rot:
            for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                avg = self._average_dir(t, rot, subjects_to_run)
                dfs.append(avg)
        else:
            for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
                avg = self._average_dir(t, r, subjects_to_run)
                dfs.append(avg)

        self.averages = dfs
        return dfs

    def filter_data(self, window_size=15):
        """
        Runs moving average on data stored inside object
        """
        for key in self.data.keys():
            for t in self.data[key]:
                t.moving_average(window_size)

        self.filtered = True
        self.window_size = window_size

    def save_data(self):
        """
        Saves each AsteriskTrialObject
        """
        for key in self.data.keys():
            for t in self.data[key]:
                t.save_data()
                # print(f"Saved: {t.generate_name()}")

    def _make_plot(self, trials, filtered=True, stds=False):
        """
        Function to make our plots.
        :param trials either a list of AsteriskTrialData or AsteriskAverage objs
        """
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive",
                  "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

        # plot data
        for i, t in enumerate(trials):
            data_x, data_y, theta = t.get_poses(filtered)
            plt.plot(data_x, data_y, color=colors[i], label='trajectory')

            if stds:
                t.plot_sd(colors[i])

            # # plot data points separately to show angle error with marker size # TODO: doesn't work, not sure why
            # for n in range(len(data_x)):
            #     plt.plot(data_x[n], data_y[n], color=colors[i], alpha=0.5, markersize=10 * theta[n])

        # plot target lines as dotted lines
        self.plot_all_target_lines(colors)  # TODO: maybe break out as a parameter?
        plt.xticks(np.linspace(-0.6, 0.6, 11), rotation=30)
        plt.yticks(np.linspace(-0.6, 0.6, 11))

        return plt

    def plot_data_subset(self, subject_to_run, trial_number="1", show_plot=True, save_plot=False):
        """
        Plots a subset of the data, as specified in parameters
        """

        dfs = self._get_ast_batch([subject_to_run], trial_number)  # TODO: make this work for the hand data object

        plt = self._make_plot(dfs)
        plt.title(f"Plot: {subject_to_run}_{self.hand.get_name()}, set #{trial_number}")

        if show_plot:
            plt.show()

        if save_plot:
            plt.savefig(f"pics/fullplot4_{subject_to_run}_{self.hand.get_name()}_{trial_number}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

    def plot_avg_data(self, subjects_to_run=None, r="n", show_plot=True, save_plot=False):
        """
        Plots the data from one subject, averaging all of the data in each direction
        """
        # dfs = []
        # for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        #     avg = self._average_dir(t, r, subjects_to_run)
        #     dfs.append(avg)
        avgs = self.calc_all_avgs()

        plt = self._make_plot(avgs, filtered=False, stds=True)
        if subjects_to_run:
            plt.title(f"{subjects_to_run}, {self.hand.get_name()}, {r}")
        else:
            plt.title(f"Averaged {self.hand.get_name()}, {r}")

        if save_plot:
            if subjects_to_run:
                plt.savefig(f"pics/avgplot4_{subjects_to_run}_{r}_{self.hand.get_name()}.jpg", format='jpg')
            else:
                plt.savefig(f"pics/avgplot4_all_{r}_{self.hand.get_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.show()

    def plot_orientation_error(self):
        """
        line plot of orientation error throughout a trial
        """
        pass

    def plot_fd_set(self, subject_to_run, trial_number="1", show_plot=True, save_plot=False):
        """
        plots a subset of data
        """
        trials = self._get_ast_batch(subject_to_run)
        # dirs = ["a", "b", "c", "d", "e", "f", "g", "h"]  # TODO: add cw and ccw later

        for i, t in enumerate(trials):
            # plot the fd values for that direction
            # TODO: currently only for "n", but that's a limitation of _get_ast_batch right now
            trial_label = f"{t.trial_translation}_{t.trial_rotation}"
            t_fd = t.translation_fd
            r_fd = t.rotation_fd

            plt.bar(i, t_fd, trial_label)

            # TODO: finish this


        pass

    def plot_avg_fd(self, subjects_to_run=None, r="n", show_plot=True, save_plot=False):
        """
        plots averaged fd values in a bar chart
        """
        pass

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
    h.plot_avg_data(subjects_to_run="sub1", r="n", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects_to_run="sub1", r="m15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects_to_run="sub1", r="p15", show_plot=False, save_plot=True)
    plt.clf()

    # subject 2 averages
    h.plot_avg_data(subjects_to_run="sub2", r="n", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects_to_run="sub2", r="m15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects_to_run="sub2", r="p15", show_plot=False, save_plot=True)
    plt.clf()

    # all subjects
    h.plot_avg_data(subjects_to_run=None, r="p15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects_to_run=None, r="m15", show_plot=False, save_plot=True)
    plt.clf()
    h.plot_avg_data(subjects_to_run=None, r="n", show_plot=True, save_plot=True)


