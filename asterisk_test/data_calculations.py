import similaritymeasures as sm
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from data_plotting import AsteriskPlotting


class AsteriskCalculations:
    rotations = {"a": 270, "b": 315, "c": 0, "d": 45, "e": 90,
                 "f": 135, "g": 180, "h": 225, "n": 0,
                 "no": 270, "ne": 315, "ea": 0, "se": 45, "so": 90,
                 "sw": 135, "we": 180, "nw": 225, "x": 0
                 }

    def __init__(self):
        pass

    @staticmethod
    def points_within_bounds_df(points, x_val, bounds):
        """
        Function which gets all the points that fall in a specific value range in a dataframe
        :param points: list of all points to sort
        :param x_val: x value to look around
        :param bounds: bounds around x value to look around
        """
        hi_val = x_val + bounds
        lo_val = x_val - bounds

        #print(f"t_pose: {x_val} +/- {bounds}")

        points_in_bounds = points[(points['x'] >= lo_val) & (points['x'] <= hi_val)]

        return points_in_bounds

    @staticmethod
    def points_within_bounds_list(points, x_val, bounds):
        """
        Get points in a list that fall within the bounds
        """
        hi_val = x_val + bounds
        lo_val = x_val - bounds
        points_in_bounds = []

        for point in points:
            p_x = point[0]
            p_y = point[1]

            if lo_val <= p_x <= hi_val:
                points_in_bounds.append([p_x, p_y])

        #pdb.set_trace()

        #print(np.asarray(points_in_bounds))
        return np.asarray(points_in_bounds)

    @staticmethod
    def t_distance(pose1, pose2):
        """
        Euclidean distance between two poses
        """
        return np.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)

    @staticmethod
    def r_distance(pose1, pose2):
        """
        Rotational distance between two poses
        """
        # TODO: do I even need this? Its just two subtracting the two rotations right?
        return pose1[2] - pose2[2]

    @staticmethod
    def angle_between(pose1, pose2):
        """
        Calculates angle between two poses as if they were lines from the origin. Uses the determinant.
        :param pose1:
        :param pose2:
        :return:
        """
        M_dot = np.dot(pose1, pose2)
        mag_t = np.sqrt(pose1[0]**2 + pose1[1]**2)
        mag_pt = np.sqrt(pose2[0]**2 + pose2[1]**2)
        mags = mag_t * mag_pt
        rad = np.arccos(M_dot / mags)
        angle = np.rad2deg(rad)
        return angle

    @staticmethod
    def narrow_target(obj_pose, target_poses, scl_ratio=(0.5, 0.5)) -> int:
        """ narrow down the closest point on the target poses
        :param obj_pose last object pose Pose2D
        :param target_poses [Pose2D;
        :param scl_ratio - how much to scale distance and rotation error by
        :returns target_i the index of the best match """

        dists_targets = [AsteriskCalculations.t_distance(obj_pose, p) for p in target_poses]
        i_target = dists_targets.index(min(dists_targets))

        return i_target

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

        return [new_x, new_y]

    @staticmethod
    def rotate_points(points, ang, use_filtered=False):
        """
        Rotate points so they are horizontal, used in averaging
        :param points: points is a dataframe with 'x', 'y', 'rmag' columns
        :param ang: angle to rotate data
        """
        # TODO: make more efficient with apply function or something?
        rad = np.radians(ang)
        rotated_line = pd.DataFrame(columns=['x', 'y', 'rmag'])

        for p in points.iterrows():
            if use_filtered:
                x = p[1]['f_x']
                y = p[1]['f_y']
            else:
                x = p[1]['x']
                y = p[1]['y']

            new_x = x*np.cos(rad) - y*np.sin(rad)
            new_y = y*np.cos(rad) + x*np.sin(rad)
            rotated_line = rotated_line.append({"x": new_x, "y": new_y, "rmag": p[1]['rmag']}, ignore_index=True)

        return rotated_line

    @staticmethod
    def debug_rotation(ast_trial):
        """
        Plot what the values look like when rotated, debugging tool
        """
        try:
            if ast_trial.is_ast_trial:
                rotated_points = AsteriskCalculations.rotate_points(ast_trial.poses,
                                                                    AsteriskCalculations.rotations[ast_trial.trial_translation])
            else:
                rotated_points = ast_trial
        except:
            rotated_points = ast_trial

        # plot the points
        x = pd.Series.to_list(rotated_points["x"].dropna())
        y = pd.Series.to_list(rotated_points["y"].dropna())

        plt.plot(x, y, color="xkcd:dark orange")

        # plot the target line
        tar_x, tar_y = AsteriskPlotting.get_c(100)
        plt.plot(tar_x, tar_y, color="xkcd:dark blue")

        # show plot
        plt.show()

    @staticmethod
    def interpolate_point(point_1, point_2, bound_val):
        """
        Linearly interpolate what the y value is at bound between the two points
        point 1 and point 2 are pandas series
        """
        # calculate slope
        slope_x = point_1['x'] - point_2['x']
        slope_y = point_1['y'] - point_2['y']
        slope = slope_y / slope_x

        # calculate y-intercept
        b_line = point_1['y'] - point_1['x'] * slope
        new_y = slope * bound_val + b_line
        new_y_index = new_y.index[0]
        # TODO: for right now, gonna ignore interpolating rmag
        interpolated_value = pd.Series({'x': bound_val, 'y': new_y.loc[new_y_index], 'rmag': None})

        return interpolated_value

    @staticmethod
    def interpolate_points(points, x_center, bounded_points, bound_size, target_points):
        #print(f"Failed to calculate area at {x_center}. Trying to interpolate values.")

        # in case the next indices are not sequential in the dataframe
        indices = points.index.to_list()
        current_index = bounded_points.index[0]
        loc_in_indices = indices.index(current_index)

        #spdb.set_trace()

        try:
            # get lower bound val
            lower_index = loc_in_indices - 1
            lower_val = AsteriskCalculations.interpolate_point(bounded_points, points.iloc[lower_index],
                                                               x_center - bound_size)
            bounded_points = bounded_points.append(lower_val, ignore_index=True)
            # plt.plot(lower_val['x'], lower_val['y'], color="r", marker='o', fillstyle='none')

        except Exception as e:
            print("low failed")
            print(e)
            print("")

        try:
            # get upper bound val
            higher_index = loc_in_indices + 1
            higher_val = AsteriskCalculations.interpolate_point(bounded_points, points.iloc[higher_index],
                                                                x_center + bound_size)
            bounded_points = bounded_points.append(higher_val, ignore_index=True)
            # plt.plot(higher_val['x'], higher_val['y'], color="r", marker='o', fillstyle='none')

        except Exception as e:
            print("high failed")
            print(e)
            print("")

        try:
            b_x = pd.Series.to_list(bounded_points["x"].dropna())
            b_y = pd.Series.to_list(bounded_points["y"].dropna())
            bounded_points_not_df = np.column_stack((b_x, b_y))
            area_calculated = sm.area_between_two_curves(bounded_points_not_df, target_points)

        except:
            print("Interpolation completely failed.")
            area_calculated = 0

        # plt.axvline(x=x_center, color='r')
        # AsteriskMetrics.debug_rotation(points)
        return area_calculated
