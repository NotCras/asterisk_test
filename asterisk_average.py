
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan, radians, mean, std
import pandas as pd
import matplotlib.pyplot as plt
from asterisk_trial import AsteriskTrialData
from asterisk_plotting import AsteriskPlotting
from asterisk_metrics import AsteriskMetrics
from asterisk_calculations import AsteriskCalculations
import pdb


class AveragedTrial(AsteriskTrialData):
    rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                 "f": 135, "g": 180, "h": 225, "n": 0}

    def __init__(self):
        super(AveragedTrial, self).__init__()  # for making an empty AsteriskTrialData object

        self.subject = []
        self.names = []  # names of trials averaged
        self.averaged_trials = []  # actual AsteriskTrialData objects that were averaged
        # self.pose_average = []  # maybe just use poses
        self.pose_ad = None
        self.pose_ad_up = None
        self.pose_ad_down = None

        # just reminding that these are here
        self.total_distance = None
        self.max_error = None
        self.translation_fd = None
        self.rotation_fd = None
        self.fd = None
        self.mvt_efficiency = None
        self.arc_len = None
        self.area_btwn = None
        self.max_area_region, self.max_area_loc = None, None
        self.metrics = None

        self.total_distance_sd = None
        self.max_error_sd = None
        self.translation_fd_sd = None
        self.fd_sd = None
        self.rotation_fd_sd = None
        self.mvt_efficiency_sd = None
        self.arc_len_sd = None
        self.area_btwn_sd = None
        self.max_area_region_sd, self.max_area_loc_sd = None, None
        self.metrics_sd = None

    def get_poses_ad(self, which_set=0):
        """
        Separates poses into x, y, theta for easy plotting.
        direction is 0 for up, 1 for down
        """
        # get the poses
        if which_set == 1:
            x = self.pose_ad_up["x"]
            y = self.pose_ad_up["y"]
            twist = self.pose_ad_up["rmag"]

        elif which_set == 2:
            x = self.pose_ad_down["x"]
            y = self.pose_ad_down["y"]
            twist = self.pose_ad_down["rmag"]
        else:
            x = self.pose_ad["x"]
            y = self.pose_ad["y"]
            twist = self.pose_ad["rmag"]

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

        if lo_val <= 0:
            points_in_bounds = points[(points['x'] >= 0) & (points['x'] <= hi_val)]
        else:
            points_in_bounds = points[(points['x'] >= lo_val) & (points['x'] <= hi_val)]

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
        values = {"dist": (0, 0), "t_fd": (0, 0), "r_fd": (0, 0), "mvt_eff": (0, 0), "btwn": (0, 0), # "fd": self.fd
                  "max_err": (0, 0), "max_a_reg": (0, 0), "max_a_loc": (0, 0), "arc_len": (0,0)}
        dist_vals = []
        t_fd_vals = []
        r_fd_vals = []
        # fd_vals = []
        mvt_eff_vals = []
        btwn_vals = []
        err_vals = []
        reg_vals = []
        loc_vals = []
        arc_lens = []

        for t in self.averaged_trials:
            dist_vals.append(t.total_distance)
            t_fd_vals.append(t.translation_fd)
            r_fd_vals.append(t.rotation_fd)
            mvt_eff_vals.append(t.mvt_efficiency)
            btwn_vals.append(t.area_btwn)
            # fd_vals.append(t.fd)
            err_vals.append(t.max_error)
            reg_vals.append(t.max_area_region)
            loc_vals.append(t.max_area_loc)
            arc_lens.append(t.arc_len)

        values["dist"] = (mean(dist_vals), std(dist_vals))
        values["t_fd"] = (mean(t_fd_vals), std(t_fd_vals))
        values["r_fd"] = (mean(r_fd_vals), std(r_fd_vals))
        values["mvt_eff"] = (mean(mvt_eff_vals), std(mvt_eff_vals))
        values["btwn"] = (mean(btwn_vals), std(btwn_vals))
        # values["fd"] = (mean(fd_vals), std(fd_vals))
        values["max_err"] = (mean(err_vals), std(err_vals))
        values["max_a_reg"] = (mean(reg_vals), std(reg_vals))
        values["max_a_loc"] = (mean(loc_vals), std(loc_vals))
        values["arc_len"] = (mean(arc_lens), std(arc_lens))

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
        self.max_error = values["max_err"][0]
        self.max_error_sd = values["max_err"][1]
        # self.fd = values["fd"][0]
        # self.fd_sd =  = values["fd"][1]
        self.max_area_region = values["max_a_reg"][0]
        self.max_area_region_sd = values["max_a_reg"][1]
        self.max_area_loc = values["max_a_loc"][0]
        self.max_area_loc_sd = values["max_a_loc"][1]
        self.arc_len = values["arc_len"][0]
        self.arc_len_sd = values["arc_len"][1]

        metric_dict = {"trial": self.generate_name(),
                       "t_fd": self.translation_fd, "r_fd": self.rotation_fd,  # "fd": self.fd
                       "max_err": self.max_error, "mvt_eff": self.mvt_efficiency, "arc_len": self.arc_len,
                       "area_btwn": self.area_btwn, "max_a_reg": self.max_area_region, "max_a_loc": self.max_area_loc}

        self.metrics = pd.Series(metric_dict)

        metric_sd_dict = {"trial": self.generate_name(),
                          "t_fd_sd": self.translation_fd_sd, "r_fd_sd": self.rotation_fd_sd,  # "fd_sd": self.fd
                          "max_err_sd": self.max_error_sd, "mvt_eff_sd": self.mvt_efficiency_sd,
                          "arc_len": self.arc_len_sd, "area_btwn_sd": self.area_btwn_sd,
                          "max_a_reg_sd": self.max_area_region_sd, "max_a_loc_sd": self.max_area_loc_sd}

        self.metric_sds = pd.Series(metric_sd_dict)

        return self.total_distance, self.translation_fd, self.rotation_fd, self.mvt_efficiency, self.area_btwn

    def make_average_line(self, trials, show_rot_debug=False):
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
            data_points = data_points.append(t.poses)  # put all poses in one dataframe for easy access

        # rotate the line so we can do everything based on the x axis. Yes, I know this is hacky
        r_target_x, r_target_y = AsteriskPlotting.get_c(50)
        rotated_target_line = np.column_stack((r_target_x, r_target_y))
        rotated_data = self._rotate_points(data_points, self.rotations[self.trial_translation])

        avg_line = pd.DataFrame()
        avg_ad = pd.DataFrame()

        avg_ad_up = pd.DataFrame()
        avg_ad_down = pd.DataFrame()

        # now we go through averaging
        for i, t in enumerate(rotated_target_line):
            t_x = t[0]
            # TODO: 0.03 is arbitrary... make bounds scale with resolution of target_line?
            points = self._get_points(rotated_data, t_x, 0.03)

            averaged_point = points.mean(axis=0)  # averages each column in DataFrame
            # std_point = points.std(axis=0)  # doesn't really show up right
            # print(f"num points averaged: {len(points)}")

            # get previous point, maybe make the current one be the next one, and then grab two back
            try:
                prev_avg = avg_line.iloc[-1]
            except:
                prev_avg = pd.Series({"x": 0., "y": 0.,
                                      "rmag": 0., "tmag": 0.})

            dx_ad, dy_ad, avg_tmag, err_rmag = self.calc_point_ad(points, averaged_point, prev_avg)

            ad_point = pd.Series({"x": dx_ad, "y": dy_ad,
                                  "rmag": err_rmag.mean(axis=0), "tmag": avg_tmag})
            ad_point_up = pd.Series({"x": averaged_point['x']+dx_ad, "y": averaged_point['x']+dy_ad,
                                  "rmag": err_rmag.mean(axis=0), "tmag": avg_tmag})
            ad_point_down = pd.Series({"x": averaged_point['x']-dx_ad, "y": averaged_point['x']-dy_ad,
                                     "rmag": err_rmag.mean(axis=0), "tmag": avg_tmag})

            # if i in [95, 96, 97, 98, 99]:
            #     pdb.set_trace()
            if i == 90:
                pdb.set_trace()

            avg_line = avg_line.append(averaged_point, ignore_index=True)
            avg_ad = avg_ad.append(ad_point, ignore_index=True)

            avg_ad_up = avg_ad_up.append(ad_point_up, ignore_index=True)
            avg_ad_down = avg_ad_down.append(ad_point_down, ignore_index=True)

        # rotate everything back
        correct_avg = self._rotate_points(avg_line, -1 * self.rotations[self.trial_translation])
        correct_ad = self._rotate_points(avg_ad, -1 * self.rotations[self.trial_translation])

        correct_ad_up = self._rotate_points(avg_ad_up, -1 * self.rotations[self.trial_translation])
        correct_ad_down = self._rotate_points(avg_ad_down, -1 * self.rotations[self.trial_translation])

        self.poses = correct_avg
        self.pose_ad = correct_ad
        self.pose_ad_up = correct_ad_up
        self.pose_ad_down = correct_ad_down
        # pdb.set_trace()

        if show_rot_debug:
            print(f"poses length: {len(self.poses)}")
            print(f"poses ad length: {len(self.pose_ad)}")
            AsteriskMetrics.debug_rotation(self)

        print(f"Averaged: {self.subject}_{self.trial_translation}_{self.trial_rotation}")

        # now filter and calculate metrics
        self.moving_average()
        metric_values = self._calc_avg_metrics()

        return correct_avg, metric_values

    def calc_point_ad(self, points, averaged_point, prev_avg):
        """
        Given the points used to average a point, and the averaged point itself,
        determine the average deviation for that point
        """
        # TODO: we need to
        err_x = points['x'] - averaged_point['x']
        err_y = points['y'] - averaged_point['y']
        err_rmag = points['rmag'] - averaged_point['rmag']
        err_tmag = []

        # calculate vector magnitudes
        for x, y in zip(err_x, err_y):
            tmag = sqrt(x ** 2 + y ** 2)
            err_tmag.append(tmag)

        ad_data = pd.DataFrame({"x": err_x, "y": err_y, "rmag": err_rmag, "tmag": err_tmag})
        avg_tmag = ad_data["tmag"].mean(axis=0)
        # avg_rmag = ad_data["rmag"].mean(axis=0)
        # pdb.set_trace()

        # get calculate normal point
        # TODO: go back 2 in order to get a better approximation, but it will be one step behind

        # # this took me forever... I'm embarrassed:
        # # https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
        slope_x = averaged_point['x'] - prev_avg['x']
        slope_y = averaged_point['y'] - prev_avg['y']
        # reciprocal_slope = -1 * slope_x / slope_y
        # dx_ad = avg_tmag / sqrt(1+reciprocal_slope**2)
        # dy_ad = (avg_tmag * reciprocal_slope) / sqrt(1+reciprocal_slope**2)
        # just need to add/subtract them to the average point and it should work

        dlen = sqrt(slope_x * slope_x + slope_y * slope_y)
        dx_ad = avg_tmag * -slope_y / dlen
        dy_ad = avg_tmag * slope_x / dlen

        # if np.isnan(averaged_point['x']):
        #     pdb.set_trace()

        return dx_ad, dy_ad, avg_tmag, err_rmag

    def plot_line_contributions(self):
        """
        Plot circles where each trial stops contributing to the line average.
        """
        circle_colors = {"sub1": "xkcd:dark blue", "sub2": "xkcd:bordeaux", "sub3": "xkcd:forrest green"}

        a_x, a_y, _ = self.get_poses(use_filtered=False)
        for t in self.averaged_trials:
            last_pose = t.get_last_pose()

            subject = t.subject
            subject_color = circle_colors[subject]

            # find narrow target on average line, index of point on line closest to last pose
            index = AsteriskCalculations.narrow_target([last_pose[0], last_pose[1]], np.column_stack((a_x, a_y)))
            # plot a dot there
            # TODO: if dots are on the same place, jiggle them a little to the side so all are visible?
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
        self.plot_sd("xkcd:burnt orange", testing=False)

        self.plot_line_contributions()

        plt.title(f"Avg Debug Plot: {self.hand.get_name()}, {self.trial_translation}_{self.trial_rotation}")

        # TODO: show at least one averaging interval
        # show target line dots... comes out a little obnoxious
        # t_l_x, t_l_y = AsteriskPlotting.get_direction(self.trial_translation)
        # for tl_x, tl_y in zip(t_l_x, t_l_y):
        #     plt.plot(tl_x, tl_y, marker='o', fillstyle='none', color="k")

        if save_plot:
            plt.savefig(f"pics/avgdebug_{self.hand.get_name()}_{self.subject}_{self.trial_translation}_"
                        f"{self.trial_rotation}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.legend()
            plt.show()

    def plot_sd(self, color, use_filtered=False, testing=True):
        """
        plot the standard deviations as a confidence interval around the averaged line
        :param color: color for sd polygon, must be compatible with matplotlib.
        :param use_filtered: enables option to use filtered or unfiltered data. Defaults to False
        """
        avg_x, avg_y, _ = self.get_poses(use_filtered=use_filtered)
        if not testing:
            ad_x, ad_y, _ = self.get_poses_ad()

            # necessary for building the polygon
            r_avg_x = list(reversed(avg_x))
            r_avg_y = list(reversed(avg_y))
            r_ad_x = list(reversed(ad_x))
            r_ad_y = list(reversed(ad_y))

            # pdb.set_trace()

            poly = []
            for ax, ay, dx, dy in zip(avg_x, avg_y, ad_x, ad_y):
                pt = [ax, ay+dy]
                poly.append(pt)

            for ax, ay, dx, dy in zip(r_avg_x, r_avg_y, r_ad_x, r_ad_y):
                pt = [ax, ay-dy]
                poly.append(pt)
        else:
            ad_x_up, ad_y_up, _ = self.get_poses_ad(which_set=1)
            ad_x_down, ad_y_down, _ = self.get_poses_ad(which_set=2)

            # necessary for building the polygon
            r_ad_x = list(reversed(ad_x_down))
            r_ad_y = list(reversed(ad_y_down))

            # pdb.set_trace()

            poly = []
            for ax, ay in zip(ad_x_up, ad_y_up):
                pt = [ax, ay]
                poly.append(pt)

            for ax, ay in zip(r_ad_x, r_ad_y):
                pt = [ax, ay]
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
    avgln.make_average_line(lines, show_rot_debug=False)
    # print(avgln.metrics)
    # print(avgln.metric_sds)

    avgln.avg_debug_plot()

