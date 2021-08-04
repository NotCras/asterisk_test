"""
Several analyzer classes which combine metric data for different sets of data and exports them. [NOT DONE]
"""


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asterisk_data_manager as datamanager
from ast_hand_data import AstHandTrials
from ast_plotting import AsteriskPlotting
import pdb

from scipy import stats
from pathlib import Path


class AstDirAnalyzer:
    """
    This class takes trials in one direction and stores (and saves) the metrics together
    """
    def __init__(self, trials, avg=None):
        self.t_dir = trials[0].trial_translation
        self.r_dir = trials[0].trial_rotation
        self.hand_name = trials[0].hand.get_name()

        metric_df = pd.DataFrame()
        for t in trials:
            metric_df = metric_df.append(t.metrics, ignore_index=True)
        metric_df = metric_df.set_index("trial")
        #print(metric_df)
        self.metrics = metric_df

        # save trial objects in case we need it
        self.avg = avg
        self.trials = trials

    def save_data(self, file_name_overwrite=None):
        """
        Saves the report as a csv file
        :return:
        """
        if file_name_overwrite is None:
            new_file_name = f"results/{self.hand_name}_{self.t_dir}_{self.r_dir}_results.csv"
        else:
            new_file_name = file_name_overwrite + ".csv"

        self.metrics.to_csv(new_file_name, index=True)


class AstHandAnalyzer:
    """
    Takes a hand data object and gets all the metrics
    """
    def __init__(self, hd):
        self.hand_name = hd.hand.get_name()
        self.hand_data = hd  # keeping it around just in case

        # make a dir analyzer for each direction
        dir_analyzers = []
        complete_df = pd.DataFrame()
        for key in hd.data.keys():
            trials = hd.data[key]
            # TODO: implement average later
            analyzer = AstDirAnalyzer(trials)
            complete_df = complete_df.append(analyzer.metrics)
            dir_analyzers.append(analyzer)

        self.analyzers = dir_analyzers
        # print(complete_df)
        # complete_df = complete_df.set_index("trial")
        self.metrics = complete_df

        avg_df = pd.DataFrame()
        avg_sd_df = pd.DataFrame()
        for a in hd.averages:
            avg_df = avg_df.append(a.metrics, ignore_index=True)
            avg_sd_df = avg_sd_df.append(a.metric_sds, ignore_index=True)

        # pdb.set_trace()
        avg_df = avg_df.set_index("trial")
        avg_sd_df = avg_sd_df.set_index("trial")

        self.avg_metrics = avg_df
        self.avg_metric_sds = avg_sd_df

    def save_data(self, file_name_overwrite=None):
        """
        Saves the report as a csv file
        :return:
        """
        names = ["metrics", "avg_metrics", "metric_sds"]
        data = [self.metrics, self.avg_metrics, self.avg_metric_sds]

        for n, d in zip(names, data):
            if file_name_overwrite is None:
                new_file_name = f"results/{self.hand_name}_{n}.csv"
            else:
                new_file_name = f"{file_name_overwrite}_{n}.csv"

            d.to_csv(new_file_name, index=True)


class AstHandComparison:
    """
    This class takes in two AsteriskHandData objects and provides a direct comparison between them.
    """
    def __init__(self, hands):
        """
        A easier way to manage the quantitative results of the asterisk test data you collect.
        Manages all quantitative metrics in a pandas dataframe
        """
        # list of hand data objects to include
        self.hands = hands  # for now, only focusing on two hands per analyzer, will expand functionality later

        # everything stored as a pandas dataframe
        for h in self.hands:
            if not h.averages:  # force hand obj to calculate averages if it hasn't already
                h.calc_averages(rotation="n")  # TODO: for now, remove for later

        self.comparison_results = self.generate_results_df()

    def generate_results_df(self):
        """
        Generates a dictionary of the averaged metrics each hand has
        # TODO: maybe also grab the standard deviation and highest value as well?
        """
        metric_vals = pd.DataFrame(columns = ["hand", "translation", "rotation", "total_distance", "translation_fd",
                                              "rotation_fd", "mvt_efficiency", "area_btwn"])

        for h in self.hands:
            for avg in h.averages:
                values = pd.Series({"hand": h.hand.get_name(),
                                    "translation": avg.trial_translation, "rotation": avg.trial_rotation,
                                    "total_distance": avg.total_distance, "total_distance_sd": avg.total_distance_sd,
                                    "translation_fd": avg.translation_fd, "translation_fd_sd": avg.translation_fd_sd,
                                    "rotation_fd": avg.rotation_fd, "rotation_fd_sd": avg.rotation_fd_sd,
                                    "mvt_efficiency": avg.mvt_efficiency, "mvt_efficiency_sd": avg.mvt_efficiency_sd,
                                    "area_btwn": avg.area_btwn, "area_btwn_sd": avg.area_btwn_sd})

                metric_vals = metric_vals.append(values, ignore_index=True)

        return metric_vals

    def save_results_report(self, file_name_overwrite=None):
        """
        Saves the report as a csv file
        :return:
        """
        if file_name_overwrite is None:
            new_file_name = f"results/"

            for h in self.hands:
                new_file_name = new_file_name + f"{h.hand.get_name()}_"

            new_file_name = new_file_name + f"results.csv"
        else:
            new_file_name = file_name_overwrite + ".csv"

        self.comparison_results.to_csv(new_file_name, index=True)

    def plot_direction(self, translation, rotation="n", show_plot=True, save_plot=False):
        """
        Plot the average path for each hand data object contained in this object for a specific
        translation rotation pair.
        """
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive",
                  "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

        # grab the average values from all hands
        avg_label = f"{translation}_{rotation}"
        averages_to_plot = []
        hand_order = []

        for h in self.hands:
            hand_order.append(h.hand.get_name())

            for a in h.averages:
                a_label = f"{a.trial_translation}_{a.trial_rotation}"

                if a_label == avg_label:
                    averages_to_plot.append(a)

        for i, a in enumerate(averages_to_plot):
            a_x, a_y, _ = a.get_poses()

            plt.plot(a_x, a_y, color=colors[i], label=hand_order[i])

        # plot the straight line
        t_x, t_y = AsteriskPlotting.get_direction(translation)
        plt.plot(t_x, t_y, color="r", linestyle="dashed")

        if save_plot:
            # added the zero to guarantee that it comes first
            plt.savefig(f"results/pics/0all_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

        return plt

    def plot_asterisk(self, rotation="n", show_plot=True, save_plot=False):
        """
        Plot the entire averaged asterisks on top of each other for each hand
        :param rotation:
        :param show_plot:
        :param save_plot:
        :return:
        """
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive",
                  "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

        linestyles = ["dotted", "dashed", "dashdot"]

        # TODO: function works, but might want to tweak the colors plotted. Make one darker, one lighter
        for i, h in enumerate(self.hands):  # subjects = None makes it default to using all subjects in the average
            h.plot_ast_avg(rotation, subjects=None, show_plot=False, save_plot=False, linestyle=linestyles[i])

        self.hands[0].plot_all_target_lines(colors)

        if save_plot:
            # added the zero to guarantee that it comes first
            plt.savefig(f"results/pics/0all_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()
            plt.show()

        return plt