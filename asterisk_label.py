#!/usr/bin/env python3
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_calculations import AsteriskCalculations
from asterisk_plotting import AsteriskPlotting
from asterisk_trial import AsteriskTrialData


class AsteriskLabelling:

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
    def check_for_movement(data, threshold=10, rot_threshold=15):
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
        if magnitude > threshold or rot_magnitude > rot_threshold:
            return True, (magnitude, threshold), (rot_magnitude, rot_threshold)
        else:
            return False, (magnitude, threshold), (rot_magnitude, rot_threshold)

    @staticmethod
    def check_no_backtracking(data, translation_label, threshold=5, debug_rotation=False):
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
        rotated_df = AsteriskCalculations.rotate_points(data,
                                                        AsteriskCalculations.rotations[translation_label])

        if debug_rotation:
            # plot the points
            x = pd.Series.to_list(rotated_df["x"].dropna())
            y = pd.Series.to_list(rotated_df["y"].dropna())

            plt.plot(x, y, color="xkcd:dark orange")

            # plot the target line
            tar_x, tar_y = AsteriskPlotting.get_c(100)
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
    def check_poor_metric_performance(data):
        """
        True if poor performance
        Hold on this one, used in hand data
        """
        pass


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

    mvt_check, (tmag, mvt_threshold), (rot_magnitude, rot_threshold) = AsteriskLabelling.check_for_movement(df)
    back_check, (cumulative, back_threshold) = AsteriskLabelling.check_no_backtracking(df, trial_translation,
                                                                                       debug_rotation=True)

    print(f"{trial} -> mvt: {mvt_check}, {tmag} v {mvt_threshold} | {rot_magnitude} v {rot_threshold}")
    print(f"{trial} -> backtrack: {back_check}, {cumulative} v {back_threshold}")
