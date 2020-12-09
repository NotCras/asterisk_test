#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from asterisk_trial import hand, ast_trial


class ast_study:

    def __init__(self):
        '''
        Class used to store all asterisk data. Contains functions to run high level analysis. 
        data - list of hand asterisks
        hands - list of hands included in data
        '''

        self.data = list()
        self.hands = list()
        pass

    def import_data(self):
        '''
        Point object to a folder and it will import everything into hand asterisks and ast_trials
        '''
        pass

    def plot_hand(self):
        '''
        Plot data for a specific hand
        '''
        pass
