
import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv

class Pose2D:
    def __init__(self, in_x=0, in_y=0, in_theta=0):
        """ Pose in 2 dimensions
        :param in_x x location
        :param in_y y location
        :param in_theta rotation angle, in degrees"""

        self.x = in_x
        self.y = in_y
        self.theta = in_theta

    def distance(self, pose, scl_ratio=(0.5, 0.5)):
        """Difference between self and another pose
        :type pose: Pose2D
        :param pose another Pose2D
        :param scl_ratio - how much to scale distance and rotation error by
        :returns a number"""

        # Standard euclidean distance, scaled by sin(45)
        dist_trans = sqrt((self.x - pose.x) ** 2 + (self.y - pose.y) ** 2) / sin(pi/4)
        # set up 20 deg of error as approximately 1 unit of translation error
        ang_rot = abs(self.theta - pose.theta)
        if ang_rot > 180:
            ang_rot -= 180
        dist_rot = (ang_rot / 45.0)

        # return the average
        return scl_ratio[0] * dist_trans + scl_ratio[1] * dist_rot

    def lin_interp(self, pose, t):
        """ linearly interpolate the pose values
        :param pose pose to interpoloate to Pose2D
        :param t a number between 0 and 1
        :return the interpolated pose Pose2D"""
        return Pose2D(self.x + t * pose.x, self.y + t * pose.y, self.theta + t * pose.theta)

    def __str__(self):
        return "x : {0:.2f}, y : {1:.2f}, theta : {2}".format(self.x, self.y, self.theta)


class AsteriskCalculations:

    def __init__(self):
        """
        This object contains the calculations necessary for
        line averaging and frechet distance calculations.
        Deals primarily in Pose2D due to legacy code
        """

        pass
