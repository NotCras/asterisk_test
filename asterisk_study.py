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

    def plot_hand(self, hand_name):
        """
        Plot data for a specific hand, with averages
        """
        pass

    def plot_all_hands(self):
        """
        Make averaged distance plots for each hand (NOT frechet distance) and plot them in a big subplot array
        """
        pass

    def plot_all_fd(self):
        """
        Make averaged frechet distance plots for each hand and plot them in a big subplot array
        """
        pass
