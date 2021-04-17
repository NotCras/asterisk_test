#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_plotting import AsteriskPlotting
from asterisk_trial import AsteriskTrialData

class AsteriskGarbageCollect:

    @staticmethod
    def check_data_direction(trial_label, data):
        """
        True if data doesn't deviate too much, False if it does
        """

        # go through each point

        # determine the angle

        # check with the trial_label's angle, make sure its within +/- 50 degrees

        # return True if its ok, False if its not
        pass

    @staticmethod
    def check_for_movement(data):
        """
        True if there's sufficient movement, False if there is not
        data :: dataframe
        TODO: test this one!
        """

        # get the last point
        last_val = data.dropna().tail(1).to_numpy()[0]  # TODO: make more comprehensive?

        # calculate the magnitude for (x, y, rotation)
        magnitude = np.sqrt(last_val[0]**2 + last_val[1]**2 + last_val[2]**2)

        # need to look at how far is acceptable (starting with 1 cm)
        if magnitude > 10:
            return True
        else:
            return False

    @staticmethod
    def check_no_backtracking(data):
        """
        True if no (or little) backtracking, False is there is
        """

        # calculate the delta x between each point

        # for those in the wrong direction, see how much accumulates and how far they get

        # if the accumulation is too big, its a False, if its ok, True
        pass

    @staticmethod
    def check_poor_performance(data):
        """
        True if poor performance
        Hold on this one, used in hand data
        """
        pass

