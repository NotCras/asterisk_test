#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_plotting import AsteriskPlotting
from asterisk_trial import trial

class AsteriskGarbageCollect:

    @staticmethod
    def check_data_direction(trial_label, data):
        """
        True if data doesn't deviate too much, False if it does
        """
        pass

    @staticmethod
    def check_for_movement(data):
        """
        True if there's sufficient movement, False if there is not
        """
        pass

    @staticmethod
    def check_no_backtracking(data):
        """
        True if no (or little) backtracking, False is there is
        """
        pass

    @staticmethod
    def check_poor_performance(data):
        """
        True if poor performance
        Hold on this one, used in hand data
        """
        pass

