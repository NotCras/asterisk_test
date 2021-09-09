import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast_trial_rotation import AstTrialRotation
from ast_hand_info import HandInfo
from data_plotting import AsteriskPlotting
from data_calculations import AsteriskCalculations
import pdb


class AveragedRotation:
    """
    Class functions as storage for averaged rotation trial values
    """  # We don't inherit from AveragedTrial because we don't store a path in the conventional sense
    def __init__(self, direction, trials, do_translations=True):
        self.direction = direction
        self.averaged_trialset = trials

        self.names, self.max_rot, self.max_trans, self.max_trans_coords, self.max_trans_coords_sd = \
            None, None, None, None, None
        self.data_demographics(trials, do_translations=do_translations)

    def data_demographics(self, trials, do_translations=True):
        names = []
        max_rots = []
        max_translations = []
        max_translations_xs = []
        max_translations_ys = []

        for t in trials:
            names.append(t.generate_name())
            max_rots.append(t.total_distance)

            if do_translations:
                tmag, tmag_coords = self._get_largest_tmag(t)
                max_translations.append(tmag)
                max_translations_xs.append(tmag_coords[0])
                max_translations_ys.append(tmag_coords[1])

        self.names = names

        avg_max_rot = np.mean(max_rots)
        std_max_rot = np.std(max_rots)
        self.max_rot = (avg_max_rot, std_max_rot)

        if do_translations:
            avg_max_translation = np.mean(max_translations)
            std_max_translation = np.std(max_translations)
            self.max_trans = (avg_max_translation, std_max_translation)
            self.max_trans_coords = (np.mean(max_translations_xs), np.mean(max_translations_ys))
            self.max_trans_coords_sd = (np.std(max_translations_xs), np.std(max_translations_ys))

    def generate_name(self):
        """
        Generates the codified name of the averaged trial.
        If there are multiple translation labels or multiple rotation labels, puts down 'x' instead.
        If no handinfo object included, omits the hand name
        """  # TODO: add hand and subject attributes
        return f"{self.hand.get_name()}__n__{self.direction}__{self.subject}"

    def _get_largest_tmag(self, trial, use_filtered=True):

        path_x, path_y, _ = trial.get_poses(use_filtered=use_filtered)

        max_tmag = -1
        max_tmag_coords = (-1, -1)
        for x, y in zip(path_x, path_y):
            tmag = np.sqrt(x**2 + y**2)

            if tmag > max_tmag:
                max_tmag = tmag
                max_tmag_coords = (x, y)

        return max_tmag, max_tmag_coords

if __name__ == '__main__':
    # demo and test
    h = "2v2"
    r = "cw"
    w = 10

    test1 = AstTrialRotation(f'sub3_{h}_n_{r}_1.csv')
    test1.moving_average(window_size=w)
    test2 = AstTrialRotation(f'sub3_{h}_n_{r}_2.csv')
    test2.moving_average(window_size=w)
    test3 = AstTrialRotation(f'sub3_{h}_n_{r}_3.csv')
    test3.moving_average(window_size=w)
    test4 = AstTrialRotation(f'sub3_{h}_n_{r}_4.csv')
    test4.moving_average(window_size=w)
    test5 = AstTrialRotation(f'sub3_{h}_n_{r}_5.csv')
    test5.moving_average(window_size=w)

    lines = [
             test1,
             test2,
             test3,
             test4,
             test5,
            ]

    avgln = AveragedRotation(direction=r, trials=lines)
    avgln.calculate_avg_line(show_debug=True, calc_ad=True, use_filtered_data=True)
    print(f"names: {avgln.names}")
    print(f"averaged line: {avgln.generate_name()}")
    print(f"tot dist: {avgln.total_distance}")
    print(f"path labels: {avgln.path_labels}")
    print(f"trialset labels: {avgln.trialset_labels}")
    print(f"metrics: {avgln.metrics}")
    print(f"avg metrics: {avgln.metrics_avgd}")
