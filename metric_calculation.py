import similaritymeasures as sm
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from ast_plotting import AsteriskPlotting
from ast_calculations import AsteriskCalculations


class AstMetrics:
    def __init__(self):
        pass

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
    def calc_max_error(ast_trial, arc_length):
        """
        calculates the max error between the ast_trial path and its target line
        If everything is rotated to C direction, then error becomes the max y value
        Need arc length to be already calculated on the asterisk trial
        """
        points = AsteriskCalculations.rotate_points(ast_trial.poses,
                                                    AsteriskCalculations.rotations[ast_trial.trial_translation])
        points = points.abs()
        max_val = points['y'].max()
        return max_val / arc_length  # ast_trial.metrics["arc_len"]   # divide it by arc length to normalize the value

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
    def calc_max_area_region(ast_trial, percent_window_size=0.2):
        """
        Calculates the area of max error by sliding a window of 20% normalized length along the target line
        Seems that 10% is too small in regions of fast movement
        """
        # TODO: cheating again by rotating points... what about negative values?
        points = AsteriskCalculations.rotate_points(ast_trial.poses,
                                                    AsteriskCalculations.rotations[ast_trial.trial_translation])
        t_x, t_y = AsteriskPlotting.get_c(100)  # TODO: maybe make target_line a pandas dataframe
        targets = np.column_stack((t_x, t_y))

        # AsteriskMetrics.debug_rotation(points)

        # prepare bound size
        bound_size = 0.5 * (percent_window_size * ast_trial.total_distance)
        x_center = bound_size + 0  # just being explicit here -> x_min is 0
        x_max = 2 * bound_size
        max_area_calculated = 0
        x_center_at_max = x_center

        while x_max <= ast_trial.total_distance:
            # print(f"Now at {x_center} || {x_max}/{ast_trial.total_distance}")
            bounded_points = AsteriskCalculations.get_points_df(points, x_center, bound_size)
            b_x = pd.Series.to_list(bounded_points["x"].dropna())
            b_y = pd.Series.to_list(bounded_points["y"].dropna())
            bounded_points_not_df = np.column_stack((b_x, b_y))

            target_points = AsteriskCalculations.get_points_list(targets, x_center, bound_size)

            try:
                area_calculated = sm.area_between_two_curves(bounded_points_not_df, target_points)
            except ValueError or IndexError:
                # usually this triggers if there aren't enough points (more than one) in the window
                # if there aren't enough points, make enough points!
                try:
                    area_calculated = AsteriskCalculations.interpolate_points(points, x_center,
                                                                              bounded_points, bound_size,
                                                                              target_points)

                    print("Successful interpolation!")
                except Exception as e:
                    print("Interpolation Failed.")
                    print(e)
                    # if not points were found at all in this region, depending on bound size
                    area_calculated = 0

            x_center = x_center + 0.1 * bound_size  # want to step in 1% increments
            x_max = x_center + bound_size

            if np.abs(area_calculated) > max_area_calculated:
                max_area_calculated = np.abs(area_calculated)
                x_center_at_max = x_center

        # percentage along the target_line line that the center of max error was located
        x_center_perc = x_center_at_max / 0.5  # gives us percentage along the full target line, for easy comparing

        # x_center_at_max_r = AsteriskMetrics.rotate_point([x_center_at_max, 0],
        #                                                  -1 * AsteriskMetrics.rotations[ast_trial.trial_translation])

        # print(f"results: {max_area_calculated}, {x_center_perc}")
        return max_area_calculated, x_center_perc