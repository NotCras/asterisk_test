import similaritymeasures as sm
import numpy as np
import pandas as pd

from asterisk_plotting import AsteriskPlotting


class AsteriskMetrics:
    rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                 "f": 135, "g": 180, "h": 225, "n": 0}

    def __init__(self):
        pass

    @staticmethod
    def get_points(points, x_val, bounds):
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

    @staticmethod
    def rotate_point(point, ang):
        """
        Rotate points so they are horizontal, used in averaging
        :param point: points is a list with [x, y]
        :param ang: angle to rotate data
        """
        rad = np.radians(ang)

        x = point[0]
        y = point[1]
        new_x = x * np.cos(rad) - y * np.sin(rad)
        new_y = y * np.cos(rad) + x * np.sin(rad)

        return [new_x,new_y]

    @staticmethod
    def rotate_points(points, ang):
        """
        Rotate points so they are horizontal, used in averaging
        :param points: points is a dataframe with 'x', 'y', 'rmag' columns
        :param ang: angle to rotate data
        """
        rad = np.radians(ang)
        rotated_line = pd.DataFrame(columns=['x', 'y', 'rmag'])

        for p in points.iterrows():
            x = p[1]['x']
            y = p[1]['y']
            new_x = x*np.cos(rad) - y*np.sin(rad)
            new_y = y*np.cos(rad) + x*np.sin(rad)
            rotated_line = rotated_line.append({"x": new_x, "y": new_y, "rmag": p[1]['rmag']}, ignore_index=True)

        return rotated_line

    @staticmethod
    def calc_frechet_distance(ast_trial):
        """
        Calculate the frechet distance between self.poses and a target path
        Uses frechet distance calculation from asterisk_calculations object
        """
        o_x, o_y, o_path_ang = ast_trial.get_poses(use_filtered=False)
        o_path_t = np.column_stack((o_x, o_y))

        t_fd = sm.frechet_dist(o_path_t, ast_trial.target_line)
        r_fd = sm.frechet_dist(o_path_ang, ast_trial.target_rotation)  # just max error right now

        return t_fd, r_fd

    @staticmethod
    def calc_frechet_distance_all(ast_trial):
        """ TODO: NOT TESTED YET
        Calculate the frechet distance between self.poses and a target path, combining both translation and rotation
        Uses frechet distance calculation from asterisk_calculations object
        """
        o_x, o_y, o_path_ang = ast_trial.get_poses(use_filtered=False)
        o_path = np.column_stack((o_x, o_y, o_path_ang))

        t_rots = [ast_trial.target_rotation * len(ast_trial.target_line)]
        combined_target = np.column_stack((ast_trial.target_line, t_rots))

        fd = sm.frechet_dist(o_path, combined_target)

        return fd

    @staticmethod
    def calc_max_error(ast_trial):
        """
        calculates the max error between the ast_trial path and its target line
        If everything is rotated to C direction, then error becomes the max y value
        """
        points = AsteriskMetrics.rotate_points(ast_trial.poses, AsteriskMetrics.rotations[ast_trial.trial_translation])
        return points['y'].max()

    @staticmethod
    def calc_mvt_efficiency(ast_trial, use_filtered=True):
        """
        Calculates the efficiency of movement of the trial
        amount of translation in trial direction / arc length of path
        returns mvt_eff, arc_length
        """  # TODO only occurs with translation, add in rotation?
        total_dist_in_direction = ast_trial.total_distance
        o_x, o_y, o_path_ang = ast_trial.get_poses(use_filtered)
        o_path_t = np.column_stack((o_x, o_y))

        trial_arc_length, _ = sm.get_arc_length(o_path_t)

        return total_dist_in_direction / trial_arc_length, trial_arc_length

    @staticmethod
    def calc_area_btwn_curves(ast_trial, use_filtered=True):
        """
        Returns the area between the trial path and the target line, only with respect to translation.
        Currently returns None for non-translation trials
        """  # TODO only occurs with translation, fails for no translation trials
        o_x, o_y, o_path_ang = ast_trial.get_poses(use_filtered)
        o_path_t = np.column_stack((o_x, o_y))
        # pdb.set_trace()

        try:
            val = sm.area_between_two_curves(o_path_t, ast_trial.target_line)
        except ValueError:  # TODO: is there a better way to handle this?
            val = None

        return val

    @staticmethod
    def calc_max_area_region(ast_trial, percent_window_size=0.1):
        """
        Calculates the area of max error by sliding a window of 10% normalized length along the target line
        """
        # TODO: cheating again by rotating points
        points = AsteriskMetrics.rotate_points(ast_trial.poses, AsteriskMetrics.rotations[ast_trial.trial_translation])
        targets = AsteriskPlotting.get_c(100)

        # prepare bound size
        bound_size = 0.5 * (percent_window_size * ast_trial.total_distance)
        x_center = bound_size + 0  # just being explicit here -> x_min is 0
        x_max = 2 * bound_size
        max_area_calculated = 0
        x_center_at_max = x_center

        while x_max <= ast_trial.total_distance:
            points = AsteriskMetrics.get_points(points, x_center, bound_size)
            target_points = AsteriskMetrics.get_points(targets, x_center, bound_size)

            area_calculated = sm.area_between_two_curves(points, target_points)

            x_center = x_center + 0.1 * bound_size  # want to step in 1% increments
            x_max = x_center + bound_size

            if area_calculated > max_area_calculated:
                max_area_calculated = area_calculated
                x_center_at_max = x_center

        x_center_at_max_r = AsteriskMetrics.rotate_point([x_center_at_max, 0],
                                                         AsteriskMetrics.rotations[-1 * ast_trial.trial_translation])

        return max_area_calculated, x_center_at_max_r
