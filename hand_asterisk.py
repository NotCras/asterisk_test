#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from asterisk_trial import hand, ast_trial


class hand_asterisk:
    def __init__(self, hand):
        '''
        Class used to store asterisk trials for a specific hand. 
        hand - custom object that stores hand data

        ast_trials - list of ast_trial objects

        ast_trial_types - (a set) a running list of all of the types of asterisk trials (None, Plus15, Minus15) included in ast_trial set 
        ast_trial_directions - (a set) a running list of all the directions of asterisk trials included in ast_trial set
        '''

        self.hand = hand

        self.ast_trials = list()

        self.types = set()
        self.directions = set()     

    def add_single_trial(self, ast):
        '''
        Adds an ast_trial object to the collection
        '''
        #TODO: check if its the right hand
        #TODO: check if there is a duplicate
        self.ast_trials.append(ast)
        self.types.add(ast.trial_type)
        self.directions.add(ast.direction)
    
        print("Added trial: " + ast.generate_name() + "\n") #println not being recognized for some reason?

    def import_data(self, hand):
        '''
        Point object to a folder and it will import the data for a specific hand
        '''
        pass

    def plot_trial_single(self):
        '''
        Plot a specific direction.
        '''
        pass

    def plot_trials_batch(self):
        '''
        Plot a batch of trials.
        '''
        pass

    def plot_trials_all(self):
        '''
        Plot all of the trials contained in object
        '''
        pass

    def plot_trials_all_avg(self):
        '''
        Average each direction and plot the average path and std.
        '''
        pass

    def plot_fd_all_avg(self):
        '''
        Average FD data for each direction and plot in a radar plot style
        '''

    

