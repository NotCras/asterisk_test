
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan, radians
import pandas as pd
from asterisk_trial import AsteriskTrialData
from asterisk_plotting import AsteriskPlotting
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
        :param: filt_flag Gives option to return filtered or unfiltered data
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
        """
        hi_val = x_val + bounds
        lo_val = x_val - bounds

        #print(f"t_pose: {x_val} +/- {bounds}")

        points_in_bounds = points[(points['x'] > lo_val) & (points['x'] < hi_val)]

        return points_in_bounds

    def _rotate_points(self, points, ang):
        """
        rotate points so they are horizontal
        points is a dataframe with 'x', 'y', 'rmag' columns
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
        Average the path of 2 or more AsteriskTrialObjects
        """

        # collect the asterisktrialdata objects
        self.names = []  # if rerunning an average with same object, make sure these lists are empty
        self.averaged_trials = []
        for t_n in trials:
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

        # rotate the line so we can do everything based on the x axis
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

    def avg_debug_plot(self):
        """
        Plots one specific average together with all the data that was averaged for sanity checking.
        """
        # TODO: show all target line points on plot, and show at least one averaging interval
        # plot the trials
        for i, t in enumerate(self.averaged_trials):
            # TODO: make it not use poses later
            plt.plot(t.poses['x'], t.poses['y'], label=f"trial {i}", alpha=0.2, color="xkcd:blue grey")

        # plot average
        plt.plot(self.poses['x'], self.poses['y'], label="avg", color="xkcd:burnt orange")
        self.plot_sd("xkcd:burnt orange")

    def plot_sd(self, color, filtered=False):
        """
        plot the standard deviations as a confidence interval around the averaged line
        """
        data_x, data_y, data_t = self.get_poses(filtered)
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

        # TODO: figure out correct setup later, it has something to do with the funky confidence intervals
        # if avg_trial.trial_translation in ["c", "g"]:
        #     for dx, dy, sx, sy in zip(data_x, data_y, sd_x, sd_y):
        #         pt = [dx + sx, dy + sy]
        #         poly.append(pt)
        #
        #     for dx, dy, sx, sy in zip(r_x, r_y, r_sx, r_sy):
        #     #for a, v in zip(reversed(asterisk_avg.pose_average), reversed(vec_offset)):
        #         pt = [dx - sx, dy - sy]
        #         poly.append(pt)
        #
        # elif avg_trial.trial_translation in ["a", "e"]:
        #     for dx, dy, sx, sy in zip(data_x, data_y, sd_x, sd_y):
        #         pt = [dx + sy, dy]
        #         poly.append(pt)
        #
        #     for dx, dy, sx, sy in zip(r_x, r_y, r_sx, r_sy):
        #         # for a, v in zip(reversed(asterisk_avg.pose_average), reversed(vec_offset)):
        #         pt = [dx - sy, dy]
        #         poly.append(pt)
        #
        # else:
        #     for dx, dy, sx, sy in zip(data_x, data_y, sd_x, sd_y):
        #         pt = [dx + sy, dy + sx]
        #         poly.append(pt)
        #
        #     for dx, dy, sx, sy in zip(r_x, r_y, r_sx, r_sy):
        #         # for a, v in zip(reversed(asterisk_avg.pose_average), reversed(vec_offset)):
        #         pt = [dx - sy, dy - sx]
        #         poly.append(pt)

        polyg = plt.Polygon(poly, color=color, alpha=0.4)
        plt.gca().add_patch(polyg)


if __name__ == '__main__':
    # demo and test
    test1 = AsteriskTrialData('sub1_2v2_a_n_1.csv')
    test2 = AsteriskTrialData('sub1_2v2_a_n_2.csv')
    test3 = AsteriskTrialData('sub1_2v2_a_n_3.csv')

    test4 = AsteriskTrialData('sub2_2v2_a_n_1.csv')
    test5 = AsteriskTrialData('sub2_2v2_a_n_2.csv')
    test6 = AsteriskTrialData('sub2_2v2_a_n_3.csv')

    lines = [test1, test2, test3, test4, test5, test6]

    avgln = AveragedTrial()
    avgln.make_average_line(lines)

    # make a comparison plot!
    import matplotlib.pyplot as plt
    # plot all trials
    plt.plot(test1.poses['x'], test1.poses['y'], label="test1", alpha=0.2, color="xkcd:blue grey")
    plt.plot(test2.poses['x'], test2.poses['y'], label="test2", alpha=0.2, color="xkcd:blue grey")
    plt.plot(test3.poses['x'], test3.poses['y'], label="test3", alpha=0.2, color="xkcd:blue grey")
    plt.plot(test4.poses['x'], test4.poses['y'], label="test4", alpha=0.4, color="xkcd:light grey")
    plt.plot(test5.poses['x'], test5.poses['y'], label="test5", alpha=0.4, color="xkcd:light grey")
    plt.plot(test6.poses['x'], test6.poses['y'], label="test6", alpha=0.4, color="xkcd:light grey")

    # draw average
    plt.plot(avgln.poses['x'], avgln.poses['y'], label="avg", color="xkcd:burnt orange")
    # draw std
    plt.fill_between(avgln.poses['x'],
                     (avgln.poses['y'] + avgln.pose_sd['y']),
                     (avgln.poses['y'] - avgln.pose_sd['y']),
                     alpha=0.5)

    # draw average filtered with moving average
    plt.plot(avgln.poses['f_x'], avgln.poses['f_y'], label="f_avg", color="xkcd:bright blue")

    plt.legend()
    plt.show()

