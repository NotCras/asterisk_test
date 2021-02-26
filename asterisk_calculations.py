
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv
import pdb


class AsteriskCalculations:

    def __init__(self, translation=None, rotation=None):
        """
        This object contains the calculations necessary for
        line averaging and frechet distance calculations.
        Deals primarily in Pose2D due to legacy code
        """

    @staticmethod
    def t_distance(pose1, pose2):
        """ euclidean distance between two poses
        :param pose1
        :param pose2
        """
        return sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)

    @staticmethod
    def r_distance(pose1, pose2):
        """

        """
        pass

    @staticmethod
    def narrow_target(obj_pose, target_poses, scl_ratio=(0.5, 0.5)) -> int:
        """ narrown down the closest point on the target poses
        :param obj_pose last object pose Pose2D
        :param target_poses [Pose2D;
        :param scl_ratio - how much to scale distance and rotation error by
        :returns target_i the index of the best match """

        dists_targets = [AsteriskCalculations.t_distance(obj_pose, p) for p in target_poses]
        i_target = dists_targets.index(min(dists_targets))

        return i_target
