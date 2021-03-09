
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan, radians
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_trial import AsteriskTrialData
from asterisk_plotting import AsteriskPlotting
from asterisk_calculations import AsteriskCalculations
import pdb


class AveragedTrial(AsteriskTrialData):
    rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                 "f": 135, "g": 180, "h": 225}

    def __init__(self):
        super(AveragedTrial, self).__init__()  # makes an empty AsteriskTrialData object

        self.subject = []
        self.names = []  # names of trials averaged
        self.averaged_trials = []  # actual AsteriskTrialData objects that were averaged
        # self.pose_average = []  # maybe just use poses
        self.pose_sd = None

    def get_poses_sd(self):
        """
        Separates poses into x, y, theta for easy plotting.
        """
        # get the poses
        x = self.pose_sd["x"]
        y = self.pose_sd["y"]
        twist = self.pose_sd["rmag"]

        return_x = pd.Series.to_list(x.dropna())
        return_y = pd.Series.to_list(y.dropna())
        return_twist = pd.Series.to_list(twist.dropna())

        return return_x, return_y, return_twist

    def _get_points(self, points, x_val, bounds):
        """
        Function which gets all the points that fall in a specific value range
        :param points: list of all points to sort
        :param x_val: x value to look around
        :param bounds: bounds around x value to look around
        """
        hi_val = x_val + bounds
        lo_val = x_val - bounds

        #print(f"t_pose: {x_val} +/- {bounds}")

        points_in_bounds = points[(points['x'] > lo_val) & (points['x'] < hi_val)]

        return points_in_bounds

    def _rotate_points(self, points, ang):
        """
        Rotate points so they are horizontal, used in averaging
        :param points: points is a dataframe with 'x', 'y', 'rmag' columns
        :param ang: angle to rotate data
        """
        rad = radians(ang)
        rotated_line = pd.DataFrame(columns=['x', 'y', 'rmag'])

        for p in points.iterrows():
            x = p[1]['x']
            y = p[1]['y']
            new_x = x*cos(rad) - y*sin(rad)
            new_y = y*cos(rad) + x*sin(rad)
            rotated_line = rotated_line.append({"x": new_x, "y": new_y, "rmag": p[1]['rmag']}, ignore_index=True)

        return rotated_line

    def make_average_line(self, trials):
        """
        Average the path of 2 or more AsteriskTrialObjects. Produces average and standard deviations.
        Saves this data on the object itself.
        :param trials: list of trials to average
        """

        # collect the asterisktrialdata objects
        self.names = []  # if rerunning an average with same object, make sure these lists are empty
        self.averaged_trials = []
        for t_n in trials:
            # TODO: get a list of the last obj poses, will plot them on data
            self.names.append(t_n.generate_name())
            self.averaged_trials.append(t_n)

        # first take attributes of first asterisktrialdata object and take its attributes
        trial = self.averaged_trials[0]
        self.subject = trial.subject  # TODO: add more subjects, make this a list? -> will affect other func too
        self.trial_translation = trial.trial_translation
        self.trial_rotation = trial.trial_rotation
        self.trial_num = trial.trial_num
        self.target_line = trial.target_line

        # get all the data
        data_points = pd.DataFrame()  # makes an empty dataframe
        for t in self.averaged_trials:
            data_points = data_points.append(t.poses)

        # rotate the line so we can do everything based on the x axis. Yes, I know this is hacky
        r_target_x, r_target_y = AsteriskPlotting.get_c(100)
        rotated_target_line = np.column_stack((r_target_x, r_target_y))
        rotated_data = self._rotate_points(data_points, self.rotations[self.trial_translation])

        avg_line = pd.DataFrame()
        avg_std = pd.DataFrame()

        # now we go through averaging
        for t in rotated_target_line:
            t_x = t[0]
            points = self._get_points(rotated_data, t_x, 0.05)
            # TODO: 0.05 is arbitrary... make bounds scale with resolution of target_line?
            averaged_point = points.mean(axis=0)  # averages each column in DataFrame
            std_point = points.std(axis=0)
            avg_line = avg_line.append(averaged_point, ignore_index=True)
            avg_std = avg_std.append(std_point, ignore_index=True)

        # rotate everything back
        correct_avg = self._rotate_points(avg_line, -1 * self.rotations[self.trial_translation])
        correct_std = self._rotate_points(avg_std, -1 * self.rotations[self.trial_translation])

        self.poses = correct_avg
        self.pose_sd = correct_std  # TODO: std confidence intervals don't seem to match up well, need to investigate

        print(f"Averaged: {self.subject}_{self.trial_translation}_{self.trial_rotation}")

        # now filter and run fd
        self.moving_average()
        # self.translation_fd, self.rotation_fd = self.calc_frechet_distance()  # TODO: broken for avg, investigate!

        return correct_avg

    def plot_line_contributions(self, subplot=None):
        """
        Plot circles where each trial stops contributing to the line average.
        """
        a_x, a_y, _ = self.get_poses(use_filtered=False)
        for t in self.averaged_trials:
            last_pose = t.get_last_pose()

            # TODO: test this
            # find narrow target on average line, index of point on line closest to last pose
            index = AsteriskCalculations.narrow_target([last_pose[0], last_pose[1]], np.column_stack((a_x, a_y)))
            # plot a dot there
            if subplot is None:
                plt.plot(a_x[index], a_y[index], marker='o', fillstyle='none', color="xkcd:dark blue")
            else:
                subplot.plot(a_x[index], a_y[index], marker='o', fillstyle='none', color="xkcd:dark blue")

    def avg_debug_plot(self, show_plot=True, save_plot=False):
        """
        Plots one specific average together with all the data that was averaged for sanity checking the average.
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """
        # plot the trials
        for i, t in enumerate(self.averaged_trials):
            t_x, t_y, _ = t.get_poses(use_filtered=False)
            plt.plot(t_x, t_y, label=f"trial {i}", alpha=0.5, color="xkcd:blue grey")

        # plot average
        a_x, a_y, _ = self.get_poses(use_filtered=False)
        plt.plot(a_x, a_y, label="avg", color="xkcd:burnt orange")
        self.plot_sd("xkcd:burnt orange")

        self.plot_line_contributions()

        # TODO: show all target line points on plot, and show at least one averaging interval

        if save_plot:
            plt.savefig(f"pics/avgdebug_{self.subject}_{self.hand.get_name()}_{self.trial_translation}_"
                        f"{self.trial_rotation}.jpg", format='jpg')
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
        data_x, data_y, data_t = self.get_poses(use_filtered)
        sd_x, sd_y, sd_t = self.get_poses_sd()

        # necessary for building the polygon
        r_x = list(reversed(data_x))
        r_y = list(reversed(data_y))
        r_sx = list(reversed(sd_x))
        r_sy = list(reversed(sd_y))

        poly = []
        for dx, dy, sx, sy in zip(data_x, data_y, sd_x, sd_y):
            pt = [dx + sx, dy + sy]
            poly.append(pt)

        for dx, dy, sx, sy in zip(r_x, r_y, r_sx, r_sy):
            pt = [dx - sx, dy - sy]
            poly.append(pt)

        polyg = plt.Polygon(poly, color=color, alpha=0.4)
        plt.gca().add_patch(polyg)


if __name__ == '__main__':
    # demo and test
    test1 = AsteriskTrialData('sub1_2v2_c_n_1.csv')
    test2 = AsteriskTrialData('sub1_2v2_c_n_2.csv')
    test3 = AsteriskTrialData('sub1_2v2_c_n_3.csv')

    test4 = AsteriskTrialData('sub2_2v2_c_n_1.csv')
    test5 = AsteriskTrialData('sub2_2v2_c_n_2.csv')
    test6 = AsteriskTrialData('sub2_2v2_c_n_3.csv')

    lines = [test1, test2, test3, test4, test5, test6]

    avgln = AveragedTrial()
    avgln.make_average_line(lines)

    avgln.avg_debug_plot()

