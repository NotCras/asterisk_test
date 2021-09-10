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
        self.hand = None

        self.names, self.max_rot, self.max_trans, self.max_trans_coords, self.max_trans_coords_sd = \
            None, None, None, None, None
        self.assess_trialset_labels()
        self.data_demographics(trials, do_translations=do_translations)
        self.calc_avg_metrics()

    def data_demographics(self, trials, do_translations=True):
        names = []
        hands = set()
        subjects = set()
        max_rots = []
        max_translations = []
        max_translations_xs = []
        max_translations_ys = []

        for t in trials:
            names.append(t.generate_name())
            max_rots.append(t.total_distance)
            hands.add(t.hand.get_name())
            subjects.add(t.subject)

            if do_translations:
                tmag, tmag_coords = self._get_largest_tmag(t)
                max_translations.append(tmag)
                max_translations_xs.append(tmag_coords[0])
                max_translations_ys.append(tmag_coords[1])

        self.names = names

        if len(hands) > 1:
            print("there is more than one hand here!")
            single_hand = None
        elif len(hands) == 1:  # TODO: what is the list is empty?
            single_hand = list(hands)[0]
        else:
            single_hand = ""

        self.hand = HandInfo(single_hand)
        self.subject = subjects

        avg_max_rot = np.mean(max_rots)
        std_max_rot = np.std(max_rots)
        self.max_rot = (avg_max_rot, std_max_rot)

        if do_translations:
            avg_max_translation = np.mean(max_translations)
            std_max_translation = np.std(max_translations)
            self.max_trans = (avg_max_translation, std_max_translation)
            self.max_trans_coords = (np.mean(max_translations_xs), np.mean(max_translations_ys))
            self.max_trans_coords_sd = (np.std(max_translations_xs), np.std(max_translations_ys))

    def assess_trialset_labels(self):
        """
        Collect all of the unique labels in the trialset
        """
        labels = []

        for t in self.averaged_trialset:
            for l in t.path_labels:
                labels.append(l)

        unique_labels = set(labels)
        self.trialset_labels = unique_labels
        return unique_labels

    def generate_name(self):
        """
        Generates the codified name of the averaged trial.
        If there are multiple translation labels or multiple rotation labels, puts down 'x' instead.
        If no handinfo object included, omits the hand name
        """  # TODO: add hand and subject attributes
        if self.hand is None:
            return f"Hand__n__{self.direction}__{self.subject}"
        else:
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

    # def average_metrics(self, mode=2, use_filtered=False):
    #     """
    #     Choose to average the collection of metrics contained in the data to average (mode 0) or to
    #     calculate metrics on the averaged line (mode 1), or both (mode 2).
    #     """
    #     if mode == 0:
    #         self.calc_avg_metrics(use_filtered=use_filtered) # TODO: use_filtered option doesn't work here
    #     elif mode == 1:
    #         self.update_all_metrics(use_filtered=use_filtered)
    #     elif mode == 2:
    #         self.calc_avg_metrics(use_filtered=use_filtered)  # TODO: use_filtered option doesn't work here
    #         self.update_all_metrics(use_filtered=use_filtered)
    #     else:
    #         print("Wrong mode chosen.")

    def calc_avg_metrics(self, use_filtered=False):
        """
        Calculates the average metric values
        """  # TODO: implement ability to switch between filtered and unfiltered metrics
        # go through each trial, grab relevant values and add them to sum
        # first index is the value, second is the standard deviation of the value
        values = {"dist": (0, 0), "t_fd": (0, 0), "fd": (0, 0), "mvt_eff": (0, 0), "btwn": (0, 0),  # "r_fd": (0, 0)
                  "max_err": (0, 0), "max_rot_err": (0, 0), "max_a_reg": (0, 0), "max_a_loc": (0, 0), "arc_len": (0, 0)}

        metric_names = ["dist", "t_fd", "fd", "mvt_eff", "btwn", "max_err", "max_rot_err",
                        "max_a_reg", "max_a_loc", "arc_len"]

        dist_vals = []  # TODO: I'm pretty sure I can make this more elegant. Keeping this way for now
        t_fd_vals = []
        # r_fd_vals = []
        fd_vals = []
        mvt_eff_vals = []
        btwn_vals = []
        err_vals = []
        rot_err_vals = []
        reg_vals = []
        loc_vals = []
        arc_lens = []

        for trial in self.averaged_trialset:
            metrics = trial.metrics

            if metrics is None:
                print(f"{trial.generate_name()} has no metrics.")
                continue

            dist_vals.append(metrics["dist"])
            t_fd_vals.append(metrics["t_fd"])
            # r_fd_vals.append(metrics["r_fd"])
            fd_vals.append(metrics["fd"])
            mvt_eff_vals.append(metrics["mvt_eff"])
            btwn_vals.append(metrics["area_btwn"])
            err_vals.append(metrics["max_err"])
            rot_err_vals.append(metrics["max_rot_err"])
            reg_vals.append(metrics["max_a_reg"])
            loc_vals.append(metrics["max_a_loc"])
            arc_lens.append(metrics["arc_len"])

        try:
            # for reference, need to get order right
            # ["dist", "t_fd", "fd", "mvt_eff", "btwn", "max_err", "max_rot_err", "max_a_reg", "max_a_loc", "arc_len"]
            for key, value_list in zip(metric_names, [dist_vals, t_fd_vals, fd_vals, mvt_eff_vals, btwn_vals, #r_fd_vals
                                                      err_vals, rot_err_vals, reg_vals, loc_vals, arc_lens]):
                values[key] = (np.mean(value_list), np.std(value_list))

        except Exception as e:
            print("Averaging Metrics Failed")
            print(f"Metric that failed: {key}")
            print(e)
            failed_idx = metric_names.index(key)
            failed_names = metric_names[failed_idx:]
            null_val = (0., 0.)

            # set all the keys to a null value
            for key in failed_names:
                values[key] = null_val

        metric_dict = {"trial": self.generate_name(), "dist": values["dist"][0],
                       "t_fd": values["t_fd"][0], "fd": values["fd"][0],  # "r_fd": values["r_fd"][0]
                       "max_err": values["max_err"][0], "max_rot_err": values["max_rot_err"][0],
                       "mvt_eff": values["mvt_eff"][0], "arc_len": values["arc_len"][0], "area_btwn": values["btwn"][0],
                       "max_a_reg": values["max_a_reg"][0], "max_a_loc": values["max_a_loc"][0]}

        self.metrics_avgd = pd.Series(metric_dict)

        metric_sd_dict = {"trial": self.generate_name(), "dist": values["dist"][1],
                          "t_fd": values["t_fd"][1], "fd": values["fd"][1],  # "r_fd": values["r_fd"][1]
                          "max_err": values["max_err"][1], "max_rot_err": values["max_rot_err"][1],
                          "mvt_eff": values["mvt_eff"][1], "arc_len": values["arc_len"][1], "area_btwn": values["btwn"][1],
                          "max_a_reg": values["max_a_reg"][1], "max_a_loc": values["max_a_loc"][1]}

        self.metrics_avgd_sds = pd.Series(metric_sd_dict)  # TODO: add into one pd dataframe -> value, sd?

        return self.metrics_avgd

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
#    avgln.calculate_avg_line(show_debug=True, calc_ad=True, use_filtered_data=True)
    print(f"names: {avgln.names}")
    print(f"averaged line: {avgln.generate_name()}")
    # print(f"tot dist: {avgln.total_distance}")
    # print(f"path labels: {avgln.path_labels}")
    # print(f"trialset labels: {avgln.trialset_labels}")
    # print(f"metrics: {avgln.metrics}")
    print(f"avg metrics: {avgln.metrics_avgd}")
