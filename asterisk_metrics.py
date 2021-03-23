import similaritymeasures as sm
import numpy as np


class AsteriskMetrics:
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
    def calc_max_error(ast_trial):
        """
        calculates the max error between the ast_trial path and its target line
        """
        pass

    @staticmethod
    def calc_mvt_efficiency(ast_trial, use_filtered=True):
        """
        Calculates the efficiency of movement of the trial
        amount of translation in trial direction / arc length of path
        """  # TODO only occurs with translation
        total_dist_in_direction = ast_trial.total_distance
        o_x, o_y, o_path_ang = ast_trial.get_poses(use_filtered)
        o_path_t = np.column_stack((o_x, o_y))

        trial_arc_length = sm.get_arc_length(o_path_t)

        return total_dist_in_direction / trial_arc_length[0], trial_arc_length[0]

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
    def calc_max_area_region(ast_trial):
        """
        Calculates the area of max error by sliding a window of 10% normalized length along the target line
        """
        pass


