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
    def __init__(self, hand1, hand2):

        self.hand1 = hand1
        self.hand2 = hand2

        # make sure there is averaged data on the hand asterisk trials, otherwise average them
        # then collect the averaged metrics
        pass

    def generate_results_report(self):
        pass

    def plot_direction(self):
        pass

    def plot_asterisk(self):
        pass



if __name__ == '__main__':
    study = AsteriskStudy(["sub1", "sub2"], ["2v2", "2v3", "3v3", "barrett"])
    study.plot_all_hands(rotation="n", show_plot=True, save_plot=True)
