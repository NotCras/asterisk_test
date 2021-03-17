
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan, radians, mean, std
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_trial import AsteriskTrialData
from asterisk_plotting import AsteriskPlotting
from asterisk_calculations import AsteriskCalculations
import pdb


class AveragedTrial(AsteriskTrialData):
    rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                 "f": 135, "g": 180, "h": 225, "n": 0}

    def __init__(self):
        super(AveragedTrial, self).__init__()  # makes an empty AsteriskTrialData object

        self.subject = []
        self.names = []  # names of trials averaged
        self.averaged_trials = []  # actual AsteriskTrialData objects that were averaged
        # self.pose_average = []  # maybe just use poses
        self.pose_ad_up = None
        self.pose_ad_down = None

        # just reminding that these are here
        self.total_distance = None
        self.translation_fd = None
        self.rotation_fd = None
        self.mvt_efficiency = None
        self.area_btwn = None

        self.total_distance_sd = None
        self.translation_fd_sd = None
        self.rotation_fd_sd = None
        self.mvt_efficiency_sd = None
        self.area_btwn_sd = None

    def get_poses_ad(self, direction=0):
        """
        Separates poses into x, y, theta for easy plotting.
        direction is 0 for up, 1 for down
        """
        # get the poses
        if direction == 0:
            x = self.pose_ad_up["x"]
            y = self.pose_ad_up["y"]
            twist = self.pose_ad_up["rmag"]
        elif direction == 1:
            x = self.pose_ad_down["x"]
            y = self.pose_ad_down["y"]
            twist = self.pose_ad_down["rmag"]

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

    def _calc_avg_metrics(self):
        """
        Calculates the average metric values
        """  # TODO: do we want to analyze on filtered or unfiltered data here? Should we force it to be one way?
        # go through each trial, grab relevant values and add them to sum
        # first index is the value, second is the standard deviation of the value
        values = {"dist": (0, 0), "t_fd": (0, 0), "r_fd": (0, 0), "mvt_eff": (0, 0), "btwn": (0, 0)}
        dist_vals = []
        t_fd_vals = []
        r_fd_vals = []
        mvt_eff_vals = []
        btwn_vals = []

        for t in self.averaged_trials:  # TODO: get standard deviations of these metrics
            dist_vals.append(t.total_distance)
            t_fd_vals.append(t.translation_fd)
            r_fd_vals.append(t.rotation_fd)
            mvt_eff_vals.append(t.mvt_efficiency)
            btwn_vals.append(t.area_btwn)

        values["dist"] = (mean(dist_vals), std(dist_vals))
        values["t_fd"] = (mean(t_fd_vals), std(t_fd_vals))
        values["r_fd"] = (mean(r_fd_vals), std(r_fd_vals))
        values["mvt_eff"] = (mean(mvt_eff_vals), std(mvt_eff_vals))
        values["btwn"] = (mean(btwn_vals), std(btwn_vals))

        self.total_distance = values["dist"][0]
        self.total_distance_sd = values["dist"][1]
        self.translation_fd = values["t_fd"][0]
        self.translation_fd_sd = values["t_fd"][1]
        self.rotation_fd = values["r_fd"][0]
        self.rotation_fd_sd = values["r_fd"][1]
        self.mvt_efficiency = values["mvt_eff"][0]
        self.mvt_efficiency_sd = values["mvt_eff"][1]
        self.area_btwn = values["btwn"][0]
        self.area_btwn_sd = values["btwn"][1]

        return self.total_distance, self.translation_fd, self.rotation_fd, self.mvt_efficiency, self.area_btwn

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
            self.names.append(t_n.generate_name())
            self.averaged_trials.append(t_n)

        # first take attributes of first asterisktrialdata object and take its attributes
        trial = self.averaged_trials[0]
        self.subject = trial.subject  # TODO: add more subjects, make this a list? -> will affect other func too
        self.hand = trial.hand
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
        # TODO: do it out without rotating?

        avg_line = pd.DataFrame()
        avg_ad_up = pd.DataFrame()
        avg_ad_down = pd.DataFrame()

        # now we go through averaging
        for t in rotated_target_line:
            t_x = t[0]
            # TODO: 0.05 is arbitrary... make bounds scale with resolution of target_line?
            points = self._get_points(rotated_data, t_x, 0.05)

            averaged_point = points.mean(axis=0)  # averages each column in DataFrame

            # average deviation -> get y coordinate errors for each point, average that, and that's what you should get
            # err_x = points['x'] - averaged_point['x']  # TODO: do I also need to do x average deviation?
            err_y = points['y'] - averaged_point['y']
            err_rmag = points['rmag'] - averaged_point['rmag']

            ad_point_up = pd.Series({"x": averaged_point['x'], "y": None, "rmag": None})
            ad_point_down = pd.Series({"x": averaged_point['x'], "y": None, "rmag": None})

            err_y_up = err_y[err_y >= averaged_point['y']]
            err_y_down = err_y[err_y < averaged_point['y']]  # this doesn't work

            ad_point_up['y'] = err_y_up.mean(axis=0) + averaged_point['y']
            ad_point_down['y'] = err_y_down.mean(axis=0) + averaged_point['y']

            ad_point_up["rmag"] = err_rmag.mean(axis=0) 
            ad_point_down["rmag"] = err_rmag.mean(axis=0)
            # std_point = points.std(axis=0)

            print(f"num points averaged: {len(points)}, num up: {len(err_y_up)}, num down: {len(err_y_down)}")

            avg_line = avg_line.append(averaged_point, ignore_index=True)
            avg_ad_up = avg_ad_up.append(ad_point_up, ignore_index=True)
            avg_ad_down = avg_ad_down.append(ad_point_down, ignore_index=True)

        # rotate everything back
        correct_avg = self._rotate_points(avg_line, -1 * self.rotations[self.trial_translation])
        correct_ad_up = self._rotate_points(avg_ad_up, -1 * self.rotations[self.trial_translation])
        correct_ad_down = self._rotate_points(avg_ad_down, -1 * self.rotations[self.trial_translation])

        self.poses = correct_avg
        self.pose_ad_up = correct_ad_up
        self.pose_ad_down = correct_ad_down

        print(f"Averaged: {self.subject}_{self.trial_translation}_{self.trial_rotation}")

        # now filter and calculate metrics
        self.moving_average()
        metric_values = self._calc_avg_metrics()

        return correct_avg, metric_values

    def plot_line_contributions(self):
        """
        Plot circles where each trial stops contributing to the line average.
        """
        circle_colors = {"sub1":"xkcd:dark blue", "sub2":"xkcd:dark lavender", "sub3":"xkcd:forrest green"}

        a_x, a_y, _ = self.get_poses(use_filtered=False)
        for t in self.averaged_trials:
            last_pose = t.get_last_pose()

            subject = t.subject
            subject_color = circle_colors[subject]

            # find narrow target on average line, index of point on line closest to last pose
            index = AsteriskCalculations.narrow_target([last_pose[0], last_pose[1]], np.column_stack((a_x, a_y)))
            # plot a dot there
            plt.plot(a_x[index], a_y[index], marker='o', fillstyle='none', color=subject_color)

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

        plt.title(f"Avg Debug Plot: {self.hand.get_name()}, {self.trial_translation}_{self.trial_rotation}")

        # TODO: show all target line points on plot, and show at least one averaging interval

        if save_plot:
            plt.savefig(f"pics/avgdebug_{self.hand.get_name()}_{self.subject}_{self.trial_translation}_"
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
        sd_x_up, sd_y_up, sd_t = self.get_poses_ad(direction=0)  # the points are already made in relation to the data
        sd_x_down, sd_y_down, _ = self.get_poses_ad(direction=1)

        # necessary for building the polygon
        r_sx_down = list(reversed(sd_x_down))
        r_sy_down = list(reversed(sd_y_down))

        poly = []
        for sx, sy in zip(sd_x_up, sd_y_up):
            pt = [sx, sy]
            poly.append(pt)

        for sx, sy in zip(r_sx_down, r_sy_down):
            pt = [sx, sy]
            poly.append(pt)

        polyg = plt.Polygon(poly, color=color, alpha=0.4)
        plt.gca().add_patch(polyg)


if __name__ == '__main__':
    # demo and test
    test1 = AsteriskTrialData('sub1_2v3_c_n_1.csv')
    test2 = AsteriskTrialData('sub1_2v3_c_n_2.csv')
    test3 = AsteriskTrialData('sub1_2v3_c_n_3.csv')

    test4 = AsteriskTrialData('sub2_2v3_c_n_1.csv')
    test5 = AsteriskTrialData('sub2_2v3_c_n_2.csv')
    test6 = AsteriskTrialData('sub2_2v3_c_n_3.csv')

    lines = [test1, test2, test3, test4, test5, test6]

    avgln = AveragedTrial()
    avgln.make_average_line(lines)

    avgln.avg_debug_plot()

