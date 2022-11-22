#!/usr/bin/env python3

"""
File for assessing labels to put on an asterisk trial object. [NOT DONE]
"""
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_calculations import AsteriskCalculations as acalc
from metric_calculation import AstMetrics as am
from data_plotting import AsteriskPlotting as aplt


class AsteriskLabelling:  # TODO: add in rotational variants to metrics to work with cw/ccw trials

    @staticmethod
    def assess_distances(ast_trial, use_filtered=True):
        """
        Assess how close the arc len and distance projections are to look for backtracking
        """
        # todo: need to add rotate points
        # TODO: need to make rotate_points more efficient... use apply or something?
        test_df = ast_trial.poses

        rotated = acalc.rotate_points(ast_trial.poses, acalc.rotations[ast_trial.trial_translation])

        last_point = rotated.dropna().tail(1).to_numpy()[0]
        dm = acalc.t_distance([last_point[0], last_point[1]], [0, 0])
        dx = last_point[0]
        dy = last_point[1]

        _, s_check = am.calc_mvt_efficiency(ast_trial)
        s = 0
        sx = test_df["x"].abs().sum()  # TODO: or do filtered...
        sy = test_df["y"].abs().sum()

        if sy > 0.1 and dm - 0.1 <= dx <= dm + 0.1:  # TODO: ran out of time, go through decision tree more
            pass  # back and forth movement
        elif sy > 0.1:
            pass  # significant backtracking

        if sx > 2*dx:
            pass  # significant backtracking

        pass

    @staticmethod
    def assess_initial_position(ast_trial, threshold=0.05, to_check=10):
        """
        Checks that trial data starts at the center, within a circle around the center pose
        """
        # check the first 50 points for falling within the initial position
        path_x, path_y, _ = ast_trial.get_poses()
        observation = False
        data_length = len(path_x)

        if to_check > data_length:
            to_check = data_length-1

        # if we don't have a point that falls within our threshold range
        # in the number of points to check, there is a problem
        for n in range(to_check):
            pt_x = path_x[n]
            pt_y = path_y[n]

            if pt_x < threshold and pt_y < threshold:
                observation = True

        return observation

    @staticmethod
    def assess_path_deviation(ast_trial, threshold=0.2, dev_perc=0.1, perc_tail_check=0.1,
                              rotation_threshold=None, use_filtered=False):
        """
        Assesses how well path remains in within a threshold
        """

        rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                     "f": 135, "g": 180, "h": 225, "n": 0,
                     "no": 270, "ne": 315, "ea": 0, "se": 45, "so": 90,
                     "sw": 135, "we": 180, "nw": 225, "x": 0
                     }

        path = ast_trial.poses

        # first, we rotate the line to C
        rotated_line = acalc.rotate_points(path, rotations[ast_trial.trial_translation], use_filtered=use_filtered)

        # next, we go through the y values and find if there are any which are outside the threshold
        #rotated_line.loc[ abs(rotated_line["y"]) > threshold, "is_dev" ] = 1
        # df.loc[df['set_of_numbers'] <= 4, 'equal_or_lower_than_4?'] = 'True'
        # df.loc[df['set_of_numbers'] > 4, 'equal_or_lower_than_4?'] = 'False'
        #rotated_line.loc['is_dev'].mask(rotated_line['y'] > threshold or rotated_line['y'] < -threshold, 1, inplace=True)
        is_deviated = rotated_line["y"].apply(lambda x: 1 if x > threshold or x < -1*threshold else 0)

        num_pts = len(is_deviated)
        try:
            num_dev = is_deviated.value_counts()[1]
        except KeyError:  # this means that 1 never showed up, and that no part of the line deviated
            num_dev = 0

        perc_deviated = num_dev / num_pts
        dev_assessment = perc_deviated > dev_perc

        if rotation_threshold is not None:
            is_rot_deviated = rotated_line["rmag"].apply(lambda x: 1 if x > rotation_threshold or x < -1*rotation_threshold else 0)

            try:
                num_rot_dev = is_rot_deviated.value_counts()[1]
            except KeyError:  # this means that 1 never showed up, and that no part of the line deviated
                num_rot_dev = 0

            perc_rot_deviated = num_rot_dev / num_pts
            rot_dev_assessment = perc_rot_deviated > dev_perc
        else:
            rot_dev_assessment = None

        # also, we check the last 10% of the path to see if it ends deviated
        last_section = is_deviated.tail(int(np.ceil(num_pts*perc_tail_check)))
        too_dev_assessment = 1 in last_section.values

        # return these values!
        return dev_assessment, too_dev_assessment, rot_dev_assessment


    @staticmethod
    def assess_path_deviation_with_rotation(ast_trial_rot, threshold=0.1):
        """
        Returns percentage of points on the path that are out of bounds
        """  # TODO: get a debug function for this?
        last_target_pt = ast_trial_rot.target_line[-1]
        path_x, path_y, _ = ast_trial_rot.get_poses()

        num_pts = len(path_x)
        pts_deviated = 0
        result = False

        for x, y in zip(path_x[1:], path_y[1:]):
            if x == 0 and y == 0:  # skip points at the origin
                continue

            # for a AstTrialRotation, we don't care about the angle between b/c we (hopefully) are staying
            # in one spot. So instead, we will look at whether the
            if x > threshold or y > threshold:
                #print(f"Greater than {threshold} deviation from center detected at pt: ({x}, {y})")
                # count this towards the number of points that are out of bounds
                pts_deviated += 1
                result = True

        perc_deviated = pts_deviated / num_pts

        return result, perc_deviated

    @staticmethod
    def assess_movement(data, threshold=10, rot_threshold=15):
        """
        True if there's sufficient movement, False if there is not

        parameters
        ----------
        - data - dataframe, not normalized yet
        - threshold - threshold for movement (in mm), default is 5 mm
        - rot_threshold - threshold for rotational movement

        returns
        -------
        - decision - (True/False that there's sufficient movement)
        - (magnitude of translation, magnitude of rotation, threshold) - provided for debugging
        TODO: test this one more thoroughly!
        """
        # convert to mm
        data = data * [1., 1., 1., 1000., 1000., 1000., 1., 1000.]
        data = data.round(4)

        # get the last point
        last_val = data.dropna().tail(1)  # TODO: make more comprehensive?

        # calculate the magnitude for (x, y, rotation), can't use tmag in aruco data because that uses z val as well
        magnitude = np.sqrt(last_val["x"].to_numpy()[0]**2 + last_val["y"].to_numpy()[0]**2)
        rot_magnitude = last_val["rmag"].to_numpy()[0]

        # need to look at how far is acceptable (starting with 1 cm)
        if magnitude > threshold or rot_magnitude > rot_threshold:  # TODO: and?
            return True, (magnitude, threshold), (rot_magnitude, rot_threshold)
        else:
            return False, (magnitude, threshold), (rot_magnitude, rot_threshold)

    @staticmethod
    def assess_path_movement(ast_trial, backtrack_threshold=0.1, shuttling_threshold=2.0, use_filtered=False):
        """
        Assess if there is backtracking (significant negative progress in target direction)
        or shuttling (significant movement mostly-orthogonal to target direction).
        Will only work for AstTrials, because AveragedTrial averaging erases this data.

        Returns a list of labels
        """
        observations = []  # TODO: make this a set to reduce duplicates? Or maybe I can use duplicates to assess more...
        if ast_trial.is_avg_trial():
            print("Data is averaged, backtrack and shuttling assessment does not apply.")
            return observations

        # rotate data to C
        rotated_data = acalc.rotate_points(ast_trial.poses, acalc.rotations[ast_trial.trial_translation],
                                           use_filtered=use_filtered)

        prev_row = None
        # pd.Series({'x':0, 'y':0, 'rmag':0})
        backtrack_accumulator = 0

        # go through each point
        for row in rotated_data.iterrows():

            if prev_row is None:
                dx = row[1]['x'] - 0
                dy = row[1]['y'] - 0

            else:
                dx = row[1]['x'] - prev_row[1]['x']
                dy = row[1]['y'] - prev_row[1]['y']

            s = np.sqrt(dx**2 + dy**2)

            # assess for backtracking
            if dx < 0:
                backtrack_accumulator += dx

                if backtrack_accumulator >= backtrack_threshold:
                    observations.append("backtracking")
            else:
                backtrack_accumulator = 0

            # assess for shuttling
            if dx > 0:
                mvt_ratio = s / dx  # amount of path per amount of movement in target direction

                if mvt_ratio > shuttling_threshold:
                    observations.append("shuttling")  # TODO: is there any more logic I want to put into this?
            else:
                pass  # this handles a string of 0,0 points

        return observations  # TODO: need to process observations before we send them out?


    @staticmethod
    def assess_no_backtracking(data, translation_label, threshold=5, debug_rotation=False):
        """
        True if no (or little) backtracking, False is there is

        parameters
        ----------
        - data - dataframe, not normalized yet
        - threshold - how much backtracking to allow, default is 10 mm

        returns
        -------
        - decision - (True/False that there's no significant backtracking)
        - (the most backtracking, threshold) - provided for debugging,
        when True provides the highest cumulated backtrack
        TODO: test this one more thoroughly!
        """
        # convert to mm
        data = data * [1., 1., 1., 1000., 1000., 1000., 1., 1000.]
        data = data.round(4)

        # rotate everything to c direction
        rotated_df = acalc.rotate_points(data, acalc.rotations[translation_label])

        if debug_rotation:
            # plot the points
            x = pd.Series.to_list(rotated_df["x"].dropna())
            y = pd.Series.to_list(rotated_df["y"].dropna())

            plt.plot(x, y, color="xkcd:dark orange")

            # plot the target line
            tar_x, tar_y = aplt.get_c(100)
            plt.plot(tar_x*100, tar_y*100, color="xkcd:dark blue")

            # show plot
            plt.show()

        # calculate the delta x between each point
        dxs = []  # hold the delta x vals
        prev_x = 0
        for p in rotated_df.iterrows():
            x = p[1]['x']
            dx = x - prev_x
            dxs.append(dx)

            prev_x = x

        rotated_df["dx"] = dxs

        # for those in the wrong direction, see how much accumulates and how far they get
        c_val = 0  # variable which stores cumulative values
        max_c_val = 0
        for p in rotated_df.iterrows():
            if p[1]["dx"] < 0:
                c_val = c_val + p[1]["dx"]

                if c_val > threshold:
                    return False, (c_val, threshold)

                if abs(c_val) > abs(max_c_val):
                    max_c_val = c_val
            else:
                c_val = 0

            print(f"c:{c_val} for {p[1]['x']}, {p[1]['dx']}")

        # if we went through the entire path and found no significant backtracking, then True!
        return True, (c_val, threshold)

    @staticmethod
    def assess_poor_metrics(data):
        """
        True if poor performance
        Hold on this one, used in hand data
        """
        pass

    @staticmethod
    def rotate_debug(ast_trial, file_loc, use_filtered=False, show_plot=True, save_plot=False):
        rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                     "f": 135, "g": 180, "h": 225, "n": 0,
                     "no": 270, "ne": 315, "ea": 0, "se": 45, "so": 90,
                     "sw": 135, "we": 180, "nw": 225, "x": 0
                     }
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)

        aplt.plot_all_target_lines()

        path = ast_trial.poses

        # first, we rotate the line to C
        rotated_line = acalc.rotate_points(path, rotations[ast_trial.trial_translation], use_filtered=use_filtered)

        plt.plot(path["x"], path["y"])
        plt.plot(rotated_line["x"], rotated_line["y"])

        fig.suptitle(f"{ast_trial.generate_name()} Rotation Debug", fontweight="bold", fontsize=14)
        # plt.title(f"{self.hand.get_name()} avg asterisk")  # , rot: {trials[0].trial_rotation}")
        ax.axis([-0.7, 0.7, -0.7, 0.7])
        ax.tick_params(axis="x", rotation=30)
        # plt.xticks(np.linspace(-0.7, 0.7, 15), rotation=30)
        # plt.yticks(np.linspace(-0.7, 0.7, 15))
        plt.gca().set_aspect('equal', adjustable='box')

        if show_plot:
            plt.legend()
            plt.show()

        if save_plot:
            plt.savefig(file_loc.debug_figs / f"rotate_debug_{ast_trial.hand.get_name()}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")


if __name__ == "__main__":
    # run checks on a certain file
    subject = "1"
    hand = "2v2"
    trial_translation = "f"
    trial_num = "3"
    trial = f"sub{subject}_{hand}_{trial_translation}_n_{trial_num}"

    # grab data from aruco data folder
    folder_path = f"aruco_data/{trial}.csv"

    df = pd.read_csv(folder_path, skip_blank_lines=True)

    df = df.set_index("frame")

    mvt_check, (tmag, mvt_threshold), (rot_magnitude, rot_threshold) = AsteriskLabelling.assess_movement(df)
    back_check, (cumulative, back_threshold) = AsteriskLabelling.assess_no_backtracking(df, trial_translation,
                                                                                        debug_rotation=True)

    print(f"{trial} -> mvt: {mvt_check}, {tmag} v {mvt_threshold} | {rot_magnitude} v {rot_threshold}")
    print(f"{trial} -> backtrack: {back_check}, {cumulative} v {back_threshold}")
