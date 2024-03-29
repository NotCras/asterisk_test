"""
Handles averaging multiple AstTrial objects in one direction. Handles both the paths and metrics.
"""

import numpy as np
from numpy import sqrt, mean, std
import pandas as pd
import matplotlib.pyplot as plt
from ast_trial_translation import AstTrialTranslation
from ast_trial import AstTrial
from ast_hand_info import HandInfo
from data_plotting import AsteriskPlotting
from data_calculations import AsteriskCalculations
import pdb


class AveragedTranslationTrial(AstTrial):
    """
    Class handles averaging a set of AstTrial objects: averaging the path and metrics
    """
    rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                 "f": 135, "g": 180, "h": 225, "n": 0,
                 "no": 270, "ne": 315, "ea": 0, "se": 45, "so": 90,
                 "sw": 135, "we": 180, "nw": 225, "x": 0
                 }

    def __init__(self, file_obj, trials=None, sample_points=25):  # TODO make it work with non-normalized AstTrial objects?
        # super(AveragedTrial, self).__init__()  # for making an empty AsteriskTrialData object

        self.file_locs = file_obj

        self.subject = set()
        self.hand = None
        self.names = []  # names of trials averaged
        self.averaged_trialset = []  # actual AsteriskTrialData objects that were averaged
        self.poses = None
        self.pose_ad_up = None
        self.pose_ad_down = None
        self.filtered = False
        self.window_size = 0

        self.target_line, self.target_rotation = None, None

        # just reminding that these are here
        self.total_distance = None
        self.metrics = None  # metrics calculated on the averaged line
        self.metrics_avgd = None  # average metrics averaged from AstTrial objects included
        self.metrics_avgd_sds = None

        self.path_labels = []  # labels assessed specifically on the averaged line
        self.trialset_labels = []  # labels contained inside the data

        if trials is not None:
            self.averaged_trialset = trials
            self.data_demographics()
            self.calculate_avg_line(sample_points=sample_points)  # avg metrics happens inside here
            self.assess_labels(mode=2)
            self.update_all_metrics()

    def data_demographics(self, subject=None, translation=None, rotation=None, number=None, controller=None):
        """
        Get data demographics from the data to average.
        Subject, translation, rotation, and number parameters are used differently here.
        They will serve as list parameters to set the expectation of what data you have in the averaged trial.

        To make everything work, if there is more than one translation/rotation/hand it will mark it with an 'x'

        """
        names = []
        subjects = set()
        hands = set()
        translations = set()
        rotations = set()
        numbers = set()
        controllers = set()
        normalized = set()

        # if we have trials to average, then go through them and pull out the relevant info
        if self.averaged_trialset:
            for a in self.averaged_trialset:
                names.append(a.generate_name())
                subjects.add(a.subject)
                hands.add(a.hand.get_name())
                translations.add(a.trial_translation)
                rotations.add(a.trial_rotation)
                numbers.add(a.trial_num)
                controllers.add(a.controller_label)  # TODO: need to add removal of Nones in each set
                normalized.add(a.normalized)

        if not self._set_check(subject, subjects) or not self._set_check(translation, translations) or \
                not self._set_check(rotation, rotations) or not self._set_check(controller, controllers):
            print("Averaged Trial demographics do not match expected demographics")

        if len(hands) > 1:
            print("there is more than one hand here!")
            single_hand = None
        elif len(hands) == 1:  # TODO: what is the list is empty?
            single_hand = list(hands)[0]
        else:
            single_hand = ""

        if len(translations) > 1:
            print("there is more than one translation here!")
            single_t = "x"
        elif len(translations) == 1:
            single_t = list(translations)[0]
        else:
            single_t = ""

        if len(rotations) > 1:
            print("there is more than one translation here!")
            single_r = "x"
        elif len(rotations) == 1:
            single_r = list(rotations)[0]
        else:
            single_r = ""

        if len(controllers) > 1:
            print("there is more than one translation here!")
            single_c = "x"
        elif len(controllers) == 1:
            single_c = list(controllers)[0]
        else:
            single_c = ""

        if len(normalized) > 1:
            print("there's a mix of normalized and un-normalized data here!")
            # TODO: throw error?
            single_n = "x"
        else:
            single_n = list(normalized)[0]

        self.subject = subjects
        self.names = names
        self.hand = HandInfo(single_hand)
        self.trial_translation = single_t
        self.trial_rotation = single_r
        self.trial_num = numbers  # no real need for this?
        self.controller_label = single_c
        self.normalized = single_n

    def generate_name(self):
        """
        Generates the codified name of the averaged trial.
        If there are multiple translation labels or multiple rotation labels, puts down 'x' instead.
        If no handinfo object included, omits the hand name
        """
        if self.hand is None:
            return f"Hands__{self.trial_translation}__{self.trial_rotation}__{self.subject}"
        else:
            return f"{self.hand.get_name()}__{self.trial_translation}__{self.trial_rotation}__{self.subject}"

    def generate_plot_title(self):
        pass  # TODO: for later, same gist as in AstBasicData

    def is_ast_trial(self):
        return False

    def is_avg_trial(self):
        return True

    def is_rot_only_trial(self):
        return False

    def add_trial(self, trial, rerun_avg=True):
        self.averaged_trialset.append(trial)
        self.data_demographics()

        if rerun_avg:
            self.calculate_avg_line()
            self.average_metrics(mode=2)
            self.assess_path_labels()

    def add_data_by_df(self, df, name='x_x_x_x_0'):  # don't use this
        trial = AstTrialTranslation(file_name=name, data=df)  # TODO: need to figure out a better way for naming...
        self.add_trial(trial)

    def _set_check(self, expected, observed_set):
        if expected is not None and isinstance(expected, list):
            expected.sort()
            observed_list = list(observed_set)
            observed_list.sort()
            if not expected == observed_list:
                print("the subject list is not as expected")
                print(f"{expected}")
                print(" vs ")
                print(f"{observed_set}")
                return False
            else:
                return True
        else:
            # print("must be a list to enforce demographic expectation")
            return True  # its true because we won't enforce the expectation

        # how do we do hand, trial translation, trial rotation, trial num?
        # should not take target line from a trial, should get a full target line

    def _get_points(self, points, x_val, bounds, use_filtered=False):
        """ # TODO: get out of this object
        Function which gets all the points that fall in a specific value range
        :param points: list of all points to sort
        :param x_val: x value to look around
        :param bounds: bounds around x value to look around
        """
        hi_val = x_val + bounds
        lo_val = x_val - bounds

        #print(f"t_pose: {x_val} +/- {bounds}")

        # if use_filtered:
        #     if lo_val <= -0.1:
        #         points_in_bounds = points[(points['f_x'] >= 0) & (points['f_x'] <= hi_val)]
        #     else:
        #         points_in_bounds = points[(points['f_x'] >= lo_val) & (points['f_x'] <= hi_val)]
        #
        # else:

        if lo_val <= -0.1:
            points_in_bounds = points[(points['x'] >= 0) & (points['x'] <= hi_val)]
        else:
            points_in_bounds = points[(points['x'] >= lo_val) & (points['x'] <= hi_val)]

        return points_in_bounds

    def _get_prev_avgs(self, i, avg_pts, at_avg_pt):
        """
        Gets previous two points, for ad calculation. Handles all cases
        """

        # based on how I've set it up, there will always be one previous average in the line. This should work
        try:
            prev_avg = avg_pts.iloc[i - 1]

            if np.isnan(prev_avg['x']):
                prev_avg = at_avg_pt
        except IndexError:
            prev_avg = avg_pts.iloc[i]

        try:
            next_avg = avg_pts.iloc[i + 1]

            if np.isnan(next_avg['x']):
                next_avg = avg_pts.iloc[i]

        except IndexError:
            next_avg = avg_pts.iloc[i]

        return prev_avg, next_avg

    def calculate_avg_line(self, sample_points=25, show_debug=False, calc_ad=True, use_filtered_data=False):
        """
        redoing averaging so that average deviation calculations are done separately after the fact
        """
        # TODO: enable ability to average on filtered data as well

        if self.averaged_trialset:
            trials = self.averaged_trialset
        else:
            print("No trials to average")
            return None

        # get all the data
        data_points = pd.DataFrame()  # makes an empty dataframe
        for t in trials:
            if not t.filtered and use_filtered_data:  # TODO: check for correct number of sample points
                t.moving_average()

            data_points = data_points.append(t.poses)  # put all poses in one dataframe for easy access

        # rotate the line so we can do everything based on the x axis. Yes, I know this is hacky


        r_target_x, r_target_y = AsteriskPlotting.get_c(sample_points)

        if not self.normalized:
            r_target_x = r_target_x*120  # units are in millimeters, so we end up with the potential for 10 cm of movement
            r_target_y = r_target_y*120  # may need to keep an eye on this one fyi TODO

        rotated_target_line = np.column_stack((r_target_x, r_target_y))

        if self.trial_translation == "x":
            pass  # TODO: multiple translation labels breaks on the next line, if I want to actually implement that

        rotated_data = AsteriskCalculations.rotate_points(data_points, self.rotations[self.trial_translation],
                                           use_filtered=use_filtered_data)
        avg_line = pd.DataFrame()

        # otherwise, very low chance we will start at 0,0 -> we know it does if the test was set up properly
        first_avg = pd.Series({"x": 0., "y": 0., "rmag": 0.})
        avg_line = avg_line.append(first_avg, ignore_index=True)

        for i, t in enumerate(rotated_target_line):
            t_x = t[0]
            if self.normalized:
                points = self._get_points(rotated_data, t_x, 0.6 / sample_points, use_filtered=use_filtered_data)
            else:  # TODO: maybe make window size a parameter or make more elegant?
                points = self._get_points(rotated_data, t_x, 60 / sample_points, use_filtered=use_filtered_data)

            averaged_point = points.mean(axis=0)  # averages each column in DataFrame

            avg_line = avg_line.append(averaged_point, ignore_index=True)

        correct_avg = AsteriskCalculations.rotate_points(avg_line, -1 * self.rotations[self.trial_translation])
        self.poses = correct_avg

        self.target_line, self.total_distance = self.generate_target_line(100, self.normalized)  # 100 samples
        self.target_rotation = self.generate_target_rot()  # TODO: doesn't work for true cw and ccw yet

        if show_debug and not calc_ad:
            print("Showing avg debug plot without average deviations.")
            self.avg_debug_plot(with_ad=False, show_plot=True, use_filtered=use_filtered_data)

        # for now, also running avg dev calculation right here, it will also show debug plot
        if calc_ad:
            self.calculate_avg_dev(all_points=rotated_data, sample_points=sample_points,
                                   show_debug=show_debug)
            # use_filtered is not used above because we just calculated average... there won't be a filtered version
            # and I want to make it so that applying a moving average is a conscious step for the user

        #self.calc_avg_metrics()
        self.average_metrics(mode=2)

        print(f"Averaged: {self.generate_name()}")
        return correct_avg

    def calculate_avg_dev(self, all_points, sample_points=25, show_debug=False,
                          show_pt_debug=False, use_filtered_data=False, use_filtered=False):
        """
        Goes through our set of averages and calculates the average deviation of the trials for each avg point
        """  # TODO: still doesn't work right when calculating avg dev on lines that are not normalized
        if not self.averaged_trialset or self.poses is None:
            self.calculate_avg_line(use_filtered_data=use_filtered_data)

        avg_ads = pd.DataFrame()
        avg_ads_up = pd.DataFrame()
        avg_ads_down = pd.DataFrame()

        avg_pts = AsteriskCalculations.rotate_points(self.poses, self.rotations[self.trial_translation],
                                                         use_filtered=use_filtered)

        r_target_x, r_target_y = AsteriskPlotting.get_c(sample_points)

        if not self.normalized:
            r_target_x = r_target_x*100  # units are in millimeters, so we end up with the potential for 10 cm of movement
            r_target_y = r_target_y*100  # may need to keep an eye on this one fyi TODO

        rotated_target_line = np.column_stack((r_target_x, r_target_y))

        # calculate the average magnitude that each point is away from the average line
        tmags = []
        tmags.append(0)
        for i, t in enumerate(rotated_target_line):
            t_x = t[0]
            avg_tmag = self._calc_avg_tmag(avg_pts.iloc[i], all_points, t_x, sample_points=sample_points,
                                           use_filtered=use_filtered)
            tmags.append(avg_tmag)

        # go through all the averages, at each point calculate the average deviation
        for i in range(len(avg_pts)):
            avg_pt = avg_pts.iloc[i]
            # get the points
            prev_avg, next_avg = self._get_prev_avgs(i, avg_pts, avg_pt)

            dx = next_avg['x'] - prev_avg['x']
            dy = next_avg['y'] - prev_avg['y']
            dlen = sqrt(dx * dx + dy * dy)
            vx = tmags[i] * -dy / dlen
            vy = tmags[i] * dx / dlen

            # add calculated offsets to a dataframe
            if dlen < 1e-10:
                # vec_offset.append((-dy, dx))
                vec_offset = pd.Series({"x": -dy, "y": dx, "rmag": 0})  # TODO: calculate rmag later
            else:
                # vec_offset.append((vx, vy))
                vec_offset = pd.Series({"x": vx, "y": vy, "rmag": 0})  # TODO: calculate rmag later

            ad_up = pd.Series({"x": avg_pt['x'] + vec_offset['x'], "y": avg_pt['y'] + vec_offset['y'], "rmag": 0})
            ad_down = pd.Series({"x": avg_pt['x'] - vec_offset['x'], "y": avg_pt['y'] - vec_offset['y'], "rmag": 0})

            if show_pt_debug and i in [5, 10, 15, 22, 23]:
                # if i in [23]:
                #     pdb.set_trace()

                pts_at_pt = self._get_points(all_points, rotated_target_line[i-1][0], 0.5 / sample_points)

                self._debug_avg_dev(i, vec_offset['x'], vec_offset['y'], tmags[i],
                                    pts_at_pt, next_avg, avg_pt, prev_avg, ad_up, ad_down)

            avg_ads = avg_ads.append(vec_offset, ignore_index=True)
            avg_ads_up = avg_ads_up.append(ad_up, ignore_index=True)
            avg_ads_down = avg_ads_down.append(ad_down, ignore_index=True)

        correct_ads = AsteriskCalculations.rotate_points(avg_ads, -1 * self.rotations[self.trial_translation])
        correct_ads_up = AsteriskCalculations.rotate_points(avg_ads_up, -1 * self.rotations[self.trial_translation])
        correct_ads_down = AsteriskCalculations.rotate_points(avg_ads_down, -1 * self.rotations[self.trial_translation])

        self.pose_ad = correct_ads
        self.pose_ad_up = correct_ads_up
        self.pose_ad_down = correct_ads_down

        if show_debug:
            print("Showing avg debug plot with average deviations.")
            self.avg_debug_plot(with_ad=True, show_plot=True,
                                use_filtered_data=use_filtered_data, use_filtered=use_filtered)

        return avg_ads

    def _calc_avg_tmag(self, avg_point, all_points, x_center, sample_points=25, use_filtered=False):
        """
        Calculate the average error
        """
        points = self._get_points(all_points, x_center, 0.5/sample_points)

        if use_filtered:
            err_x = points['f_x'] - avg_point['x']
            err_y = points['f_y'] - avg_point['y']
            err_rmag = points['f_rmag'] - avg_point['rmag']
        else:
            err_x = points['x'] - avg_point['x']
            err_y = points['y'] - avg_point['y']
            err_rmag = points['rmag'] - avg_point['rmag']

        err_tmag = []

        # calculate vector magnitudes
        for x, y in zip(err_x, err_y):
            tmag = sqrt(x ** 2 + y ** 2)
            err_tmag.append(tmag)

        ad_data = pd.DataFrame({"x": err_x, "y": err_y, "rmag": err_rmag, "tmag": err_tmag})
        avg_tmag = ad_data["tmag"].mean(axis=0)
        return avg_tmag

    def _debug_avg_dev(self, i, dx_ad, dy_ad, avg_tmag, points,
                       next_avg, avg, prev_avg,
                       ad_point_up, ad_point_down):
        # TODO: make a subplot where we also see this point in the context of the entire rotated line
        print(f"for tmag: {avg_tmag}")
        print(f"dx: {dx_ad}, dy: {dy_ad}")
        # print(f"reciprocal slope: {rec_slope}")
        # print(f"ad_point_up: {ad_point_up}")
        # print(" ")
        # print(f"ad_point_down: {ad_point_down}")

        averages_x = [next_avg['x'], prev_avg['x']]
        averages_y = [next_avg['y'], prev_avg['y']]
        plt.scatter(points["x"], points['y'], color="xkcd:blue grey", label="averaged points", alpha=0.5)
        plt.plot(averages_x, averages_y, color="xkcd:dark red", label="averages")
        plt.scatter(next_avg['x'], next_avg['y'], color="xkcd:dark red", label="avg pt", alpha=0.4)
        plt.scatter(avg['x'], avg['y'], color="xkcd:dark red", label="avg pt", alpha=0.75)
        plt.scatter(prev_avg['x'], prev_avg['y'], color="xkcd:dark red", label="avg pt", alpha=0.4)
        plt.scatter(ad_point_up['x'], ad_point_up['y'], color="xkcd:dark blue", label="ad up", alpha=0.75)
        plt.scatter(ad_point_down['x'], ad_point_down['y'], color="xkcd:dark green", label="ad down", alpha=0.75)
        plt.title(f"Debugging the calculated points at {i}")
        plt.show()

    def assess_labels(self, mode=2):
        """
        Choose to do label assessment on the collection of AstTrials (mode 0), or on the averaged path (mode 1),
        or both (mode 2)
        """
        if mode == 0:
            self.assess_trialset_labels()
        elif mode == 1:
            self.assess_path_labels()
        elif mode == 2:
            self.assess_trialset_labels()
            self.assess_path_labels()
        else:
            print("Wrong mode chosen.")

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

    def average_metrics(self, mode=2, use_filtered=False):
        """
        Choose to average the collection of metrics contained in the data to average (mode 0) or to
        calculate metrics on the averaged line (mode 1), or both (mode 2).
        """
        if mode == 0:
            self.calc_avg_metrics(use_filtered=use_filtered) # TODO: use_filtered option doesn't work here
        elif mode == 1:
            self.update_all_metrics(use_filtered=use_filtered)
        elif mode == 2:
            self.calc_avg_metrics(use_filtered=use_filtered)  # TODO: use_filtered option doesn't work here
            self.update_all_metrics(use_filtered=use_filtered)
        else:
            print("Wrong mode chosen.")

    def calc_avg_metrics(self, use_filtered=False):
        """
        Calculates the average metric values
        """  # TODO: implement ability to switch between filtered and unfiltered metrics
        # go through each trial, grab relevant values and add them to sum
        # first index is the value, second is the standard deviation of the value
        values = {"dist": (0, 0), "t_fd": (0, 0), "fd": (0, 0), "mvt_eff": (0, 0), "btwn": (0, 0),  # "r_fd": (0, 0)
                  "max_err": (0, 0), "max_err_rot": (0, 0), "max_a_reg": (0, 0), "max_a_loc": (0, 0), "arc_len": (0, 0)}

        metric_names = ["dist", "t_fd", "fd", "mvt_eff", "btwn", "max_err", "max_err_rot",
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
            rot_err_vals.append(metrics["max_err_rot"])
            reg_vals.append(metrics["max_a_reg"])
            loc_vals.append(metrics["max_a_loc"])
            arc_lens.append(metrics["arc_len"])

        try:
            # for reference, need to get order right
            # ["dist", "t_fd", "fd", "mvt_eff", "btwn", "max_err", "max_rot_err", "max_a_reg", "max_a_loc", "arc_len"]
            for key, value_list in zip(metric_names, [dist_vals, t_fd_vals, fd_vals, mvt_eff_vals, btwn_vals, #r_fd_vals
                                                      err_vals, rot_err_vals, reg_vals, loc_vals, arc_lens]):
                values[key] = (mean(value_list), std(value_list))

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
                       "max_err": values["max_err"][0], "max_err_rot": values["max_err_rot"][0],
                       "mvt_eff": values["mvt_eff"][0], "arc_len": values["arc_len"][0], "area_btwn": values["btwn"][0],
                       "max_a_reg": values["max_a_reg"][0], "max_a_loc": values["max_a_loc"][0]}

        self.metrics_avgd = pd.Series(metric_dict)

        metric_sd_dict = {"trial": self.generate_name(), "dist": values["dist"][1],
                          "t_fd": values["t_fd"][1], "fd": values["fd"][1],  # "r_fd": values["r_fd"][1]
                          "max_err": values["max_err"][1], "max_err_rot": values["max_err_rot"][1],
                          "mvt_eff": values["mvt_eff"][1], "arc_len": values["arc_len"][1], "area_btwn": values["btwn"][1],
                          "max_a_reg": values["max_a_reg"][1], "max_a_loc": values["max_a_loc"][1]}

        self.metrics_avgd_sds = pd.Series(metric_sd_dict)  # TODO: add into one pd dataframe -> value, sd?

        return self.metrics_avgd

    def _plot_line_contributions(self):
        """
        Plot circles where each trial stops contributing to the line average.
        """
        circle_colors = {"sub1": "xkcd:dark blue", "sub2": "xkcd:bordeaux", "sub3": "xkcd:forrest green"}

        a_x, a_y, _ = self.get_poses(use_filtered=False)
        for t in self.averaged_trialset:
            last_pose = t.get_last_pose()

            subject = t.subject
            subject_color = circle_colors[subject]

            # find narrow target on average line, index of point on line closest to last pose
            index = AsteriskCalculations.narrow_target([last_pose[0], last_pose[1]], np.column_stack((a_x, a_y)))
            # plot a dot there
            # TODO: if dots are on the same place, jiggle them a little to the side so all are visible?
            plt.plot(a_x[index], a_y[index], marker='o', fillstyle='none', color=subject_color)

    def avg_debug_plot(self, with_ad=True, show_plot=True, save_plot=False, use_filtered_data=False, use_filtered=False):
        """
        Plots one specific average together with all the data that was averaged for sanity checking the average.
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)

        # plot the trials
        for i, t in enumerate(self.averaged_trialset):
            # if we averaged with filtered data, just show the filtered data
            if use_filtered_data and t.filtered:
                f_x, f_y, _ = t.get_poses(use_filtered=True)
                plt.plot(f_x, f_y, label=f"trial f_{i}", alpha=0.4, color="xkcd:blue grey") #color="xkcd:grey green")

            else:
                # otherwise, let's show everything so you can wish you averaged on filtered data :P
                t_x, t_y, _ = t.get_poses(use_filtered=False)
                plt.plot(t_x, t_y, label=f"trial {i}", alpha=0.4, color="xkcd:blue grey")

                if t.filtered:
                    f_x, f_y, _ = t.get_poses(use_filtered=True)
                    plt.plot(f_x, f_y, label=f"trial f_{i}", alpha=0.8, color="xkcd:blue grey") #color="xkcd:grey green")

        # plot average
        a_x, a_y, _ = self.get_poses(use_filtered=use_filtered)
        plt.plot(a_x, a_y, label="avg", linewidth=2, color="tab:purple") #"xkcd:burnt orange")

        if with_ad:
            AsteriskPlotting.plot_sd(ax, self, color="tab:purple") #"xkcd:burnt orange", testing=True)

        # self.plot_line_contributions()

        plt.title(f"Averaged: {self.hand.get_name()}, {self.trial_translation}, {self.trial_rotation}")
        plt.legend('', frameon=False)

        self._plot_orientations(marker_scale=25, line_length=0.015, scale=1)

        if save_plot:
            file_name = f"avgdebug_{self.hand.get_name()}_{len(self.subject)}subs_{self.trial_translation}_{self.trial_rotation}.jpg"
            plt.savefig(self.file_locs.debug_figs / file_name, format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_sd(self, color, use_filtered=False):
        """
        plot the standard deviations as a confidence interval around the averaged line
        :param color: color for sd polygon, must be compatible with matplotlib.
        :param use_filtered: enables option to use filtered or unfiltered data. Defaults to False
        """
        avg_x, avg_y, _ = self.get_poses(use_filtered=use_filtered)

        ad_x_up, ad_y_up, _ = self.get_poses_ad(which_set=1)
        ad_x_down, ad_y_down, _ = self.get_poses_ad(which_set=2)

        # necessary for building the polygon
        r_ad_x = list(reversed(ad_x_down))
        r_ad_y = list(reversed(ad_y_down))

        poly = []
        for ax, ay in zip(ad_x_up, ad_y_up):
            pt = [ax, ay]
            poly.append(pt)

        # add last point for nicer looking plot
        last_pose = self.get_last_pose()
        poly.append([last_pose[0], last_pose[1]])

        for ax, ay in zip(r_ad_x, r_ad_y):
            pt = [ax, ay]
            poly.append(pt)

        polyg = plt.Polygon(poly, color=color, alpha=0.4)
        plt.gca().add_patch(polyg)

    def get_poses_ad(self, which_set=1):
        """
        Separates poses into x, y, theta for easy plotting.
        direction is 0 for up, 1 for down
        """
        # get the poses
        if which_set == 1:
            x = self.pose_ad_up["x"]
            y = self.pose_ad_up["y"]
            twist = self.pose_ad_up["rmag"]

        else: # which_set == 2: # TODO: revisit this
            x = self.pose_ad_down["x"]
            y = self.pose_ad_down["y"]
            twist = self.pose_ad_down["rmag"]

        return_x = pd.Series.to_list(x.dropna())
        return_y = pd.Series.to_list(y.dropna())
        return_twist = pd.Series.to_list(twist.dropna())

        return return_x, return_y, return_twist


if __name__ == '__main__':
    # demo and test
    h = "2v2"
    t = "a"
    r = "n"
    w = 10
    normed = False

    test1 = AstTrialTranslation(f'sub1_{h}_{t}_{r}_1.csv', norm_data=normed)
    test1.moving_average(window_size=w)
    test2 = AstTrialTranslation(f'sub1_{h}_{t}_{r}_2.csv', norm_data=normed)
    test2.moving_average(window_size=w)
    test3 = AstTrialTranslation(f'sub1_{h}_{t}_{r}_3.csv', norm_data=normed)
    test3.moving_average(window_size=w)
    test4 = AstTrialTranslation(f'sub1_{h}_{t}_{r}_4.csv', norm_data=normed)
    test4.moving_average(window_size=w)
    test5 = AstTrialTranslation(f'sub1_{h}_{t}_{r}_5.csv', norm_data=normed)
    test5.moving_average(window_size=w)

    test6 = AstTrialTranslation(f'sub2_{h}_{t}_{r}_1.csv', norm_data=normed)
    test6.moving_average(window_size=w)
    test7 = AstTrialTranslation(f'sub2_{h}_{t}_{r}_2.csv', norm_data=normed)
    test7.moving_average(window_size=w)
    test8 = AstTrialTranslation(f'sub2_{h}_{t}_{r}_3.csv', norm_data=normed)
    test8.moving_average(window_size=w)
    test9 = AstTrialTranslation(f'sub2_{h}_{t}_{r}_4.csv', norm_data=normed)
    test9.moving_average(window_size=w)
    test10 = AstTrialTranslation(f'sub2_{h}_{t}_{r}_5.csv', norm_data=normed)
    test10.moving_average(window_size=w)

    test11 = AstTrialTranslation(f'sub3_{h}_{t}_{r}_1.csv', norm_data=normed)
    test11.moving_average(window_size=w)
    test12 = AstTrialTranslation(f'sub3_{h}_{t}_{r}_2.csv', norm_data=normed)
    test12.moving_average(window_size=w)
    test13 = AstTrialTranslation(f'sub3_{h}_{t}_{r}_3.csv', norm_data=normed)
    test13.moving_average(window_size=w)
    test14 = AstTrialTranslation(f'sub3_{h}_{t}_{r}_4.csv', norm_data=normed)
    test14.moving_average(window_size=w)
    test15 = AstTrialTranslation(f'sub3_{h}_{t}_{r}_5.csv', norm_data=normed)
    test15.moving_average(window_size=w)

    #test16 = AstTrial(f'sub1_2v2_c_n_1.csv')
    #test16.moving_average(window_size=w)

    lines = [
             test1,
             test2,
             test3,
             test4,
             test5,

             test6,
             test7,
             test8,
             test9,
             test10,

             #test16,  # including this causes it to error out, still can't handle multiple translation labels

             test11,
             test12,
             test13,
             test14,
             test15
            ]

    avgln = AveragedTranslationTrial(trials=lines)
    avgln.calculate_avg_line(show_debug=True, calc_ad=True, use_filtered_data=True)
    print(f"names: {avgln.names}")
    print(f"averaged line: {avgln.generate_name()}")
    print(f"tot dist: {avgln.total_distance}")
    print(f"path labels: {avgln.path_labels}")
    print(f"trialset labels: {avgln.trialset_labels}")
    print(f"metrics: {avgln.metrics}")
    print(f"avg metrics: {avgln.metrics_avgd}")
