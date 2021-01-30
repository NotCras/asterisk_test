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

        self.data = dict()
        #self.hands = list() #TODO: change so that its one dictionary, each hand is a key and the value is the asterisk_hand object for that hand
        pass

    def import_data(self): #TODO: take in a variable that lets you selectively import data
        '''
        Point object to a folder and it will import everything into hand asterisks and ast_trials
        '''

        #generator for names

        #-for each hand, make hand asterisk object
        #-open each csv, load in data into an asterisk trial object
        #-continue for rest

        pass

    def plot_hand(self):
        '''
        Plot data for a specific hand
        '''
        pass
