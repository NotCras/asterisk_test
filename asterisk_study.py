#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asterisk_data_manager as datamanager
from asterisk_hand_data import AsteriskHandData

from scipy import stats
from pathlib import Path


class AsteriskStudy:
    def __init__(self, subjects_to_collect, hands_to_collect):
        """
        Class used to store all asterisk data. Contains functions to run high level analysis. 
        data - list of asterisk hand data objects
        hands - list of hands included in data
        """
        self.subjects = subjects_to_collect
        self.data = self._gather_data(subjects_to_collect, hands_to_collect)

    def _gather_data(self, subjects, hands):
        """
        Returns dictionary of hand asterisk data
        """
        data_dictionary = dict()
        for h in hands:
            key = h
            data_dictionary[h] = AsteriskHandData(subjects, h)

        return data_dictionary

    def filter_data(self, window_size=15):
        """
        Filter all data stored together
        """
        for h in self.data.keys():
            self.data[h].filter_data(window_size=window_size)

    def replace_hand_data(self, hand_obj):
        """
        Delete hand data from stored data and replace with new hand data obj.
        Gets attributes of obj to delete from the obj passed in
        """
        # TODO: implement this
        pass

    def save_data(self):
        """
        Save all data. This will save data at the AsteriskTrialData level
        """
        for h in self.data.keys():
            # TODO: make a nice folder structure to help organize the data by hand
            self.data[h].save_all_data()

    def generate_comparisons(self):
        """
        Makes Ast Analyzer objects to compare each hand to each other in sets of 2
        :return:
        """
        pass

    def return_hand(self, hand_name):
        """
        Returns specified AsteriskHandData object
        """
        return self.data[hand_name]

    def plot_hand(self, hand_name):
        """
        Plot data for a specific hand, with averages
        """
        hand_data = self.return_hand(hand_name)
        hand_data.plot_avg_data()

    def plot_all_hands(self, rotation="n", show_plot=True, save_plot=False):
        """
        Make averaged distance plots for each hand (NOT frechet distance) and plot them in a big subplot array
        * * * * For now, plots all 8 hands together in subplot array -> make it generalize later
        * * * *
        """

        plt.figure(figsize=(20, 15))

        for i, h in enumerate(self.data.keys()):
            plt.subplot(2, 4, i+1)
            # get hand obj
            hd = self.data[h]
            hd.plot_avg_data(self.subjects, rotation, show_plot=False, save_plot=False)

            # TODO: figure out sharing x and y axis labels

        if save_plot:
            # added the zero to guarantee that it comes first
            plt.savefig(f"pics/0all_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()
            plt.show()


    def plot_all_fd(self):
        """
        Make averaged frechet distance plots for each hand and plot them in a big subplot array
        """
        pass

class AstAnalyzer:
    """
    This class takes in two AsteriskHandData objects and provides a direct comparison between them.
    """
    def __init__(self, hands):
        """
        A easier way to manage the quantitative results of the asterisk test data you collect.
        Manages all quantitative metrics in a pandas dataframe
        """
        # list of hand data objects to include
        self.hands = hands

        # everything stored as a pandas dataframe
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
                values = pd.Series({"hand": h.hand.get_name(), "translation": avg.trial_translation,
                    "rotation": avg.trial_rotation, "total_distance": avg.total_distance,
                    "translation_fd": avg.translation_fd, "rotation_fd": avg.rotation_fd,
                    "mvt_efficiency": avg.mvt_efficiency, "area_btwn": avg.area_btwn})

                metric_vals = metric_vals.append(values)

        return metric_vals

    def save_results_report(self, file_name_overwrite=None):
        """
        Saves the report as a csv file
        :return:
        """
        if file_name_overwrite is None:
            new_file_name = f"results/"
            for h in self.hands:
                new_file_name = new_file_name + f"{h.hand.get_name}_"

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

        for h in self.hands:
            for a in h.averages:
                a_label = f"{a.trial_translation}_{a.trial_rotation}"

                if a_label == avg_label:
                    averages_to_plot.append(a)

        for i, a in enumerate(averages_to_plot):
            a_x, a_y, _ = a.get_poses()

            plt.plot(a_x, a_y, color=colors[i])

        if save_plot:
            # added the zero to guarantee that it comes first
            plt.savefig(f"pics/0all_{rotation}.jpg", format='jpg')
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

        for h in self.hands:
            h.plot_avg_data(rotation)

        self.hands[0].plot_all_target_lines(colors)
        # TODO: is that it? How can I specify between the hands? Dotted line vs dash? Not sure

        if save_plot:
            # added the zero to guarantee that it comes first
            plt.savefig(f"pics/0all_{rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

        return plt

if __name__ == '__main__':
    study = AsteriskStudy(["sub1", "sub2"], ["2v2", "2v3", "3v3", "barrett"])
    study.plot_all_hands(rotation="n", show_plot=True, save_plot=True)
