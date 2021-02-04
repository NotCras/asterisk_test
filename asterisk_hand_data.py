#!/usr/bin/env python3

import csv
import asterisk_data_manager as datamanager
import asterisk_trial as trial
from asterisk_hand import HandObj


class AsteriskHandData:
    def __init__(self, subjects, hand_name):
        """
        Class to hold all the data pertaining to a specific hand.
        Combines data from all subjects
        """
        self.hand = HandObj(hand_name)
        self.subjects_containing = subjects
        self.data = self._organize_hand_data(subjects)

    def _organize_hand_data(self, subjects_to_get):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        """
        data_dictionary = dict()
        for t, r in datamanager.generate_t_r_pairs(self.hand.get_name()):
            key = f"{t}_{r}"
            data_dictionary[key] = self._gather_trials(subjects_to_get, t, r, [1,2,3])

        return data_dictionary

    def _gather_trials(self, subjects, translation_label, rotation_label, trials):
        """
        Goes through data and compiles data with set attributes into an AsteriskTrial objects
        """
        gathered_data = list()
        for s in subjects:  # TODO: subjects is a list, make a type recommendation
            for n in trials:
                asterisk_trial_file = f"{s}_{self.hand.get_name()}_{translation_label}_{rotation_label}_{n}.csv"
                trial_data = trial.AsteriskTrial(asterisk_trial_file)
                gathered_data.append(trial_data)

        return gathered_data

    def plot_data_subset(self, subject_to_run, trial_num):  # TODO: finish these functions
        """
        Plots a subset of the data, as specified in parameters
        """
        pass

    def plot_data_1subject(self, subject_to_run):
        """
        Plots the data from one subject, averaging all of the data in each direction
        """
        pass

    def plot_data(self):
        """
        Plots all the data contained in object
        """
        pass

test_type_name = ["Translation", "Rotation",
                  "Twist_translation", "undefined"]
translation_name = ["a", "b", "c", "d", "e", "f", "g", "h", "none"]
rotation_name = ["cw", "ccw", "none"]
twist_name = ["plus15", "minus15", "none"]
translation_angles = range(90, 90-360, -45)
twist_directions = {"Clockwise": -15, "Counterclockwise": 15}
rotation_directions = {"Clockwise": -25, "Counterclockwise": 25}
subject_nums = [1, 2, 3, 4, 5]


