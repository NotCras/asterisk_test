
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
        # pdb.set_trace()
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

    @staticmethod
    def frechet_dist(poses_obj, i_target, target_poses, scl_ratio=(0.5, 0.5)) -> (int, float):
        """ Implement Frechet distance
        :param poses_obj all the object poses np.array
        :param i_target the closest point in target_poses
        :param target_poses [Pose2D];
        :param scl_ratio - how much to scale distance and rotation error by
        :returns max of the min distance between target_poses and obj_poses """

        # Length of target curve
        n_target = min(i_target + 1, len(target_poses))
        # Length of object path
        n_object_path = poses_obj.shape[1]

        # Matrix to store data in as we calculate Frechet distance
        # Entry i,j has the cost of pairing i with j, assuming best pairings up to that point
        ca = np.zeros((n_target, n_object_path), dtype=np.float64)
        ds = np.zeros((n_target, n_object_path), dtype=np.float64)
        dsum = np.zeros((n_target, n_object_path), dtype=np.float64)
        imatch = np.zeros((n_target, n_object_path), dtype=np.int)

        print("Frechet debug: target {0}, n {1}".format(i_target, n_object_path))

        # Top left corner
        ca[0, 0] = target_poses[0].distance(Pose2D(poses_obj[0, 0], poses_obj[1, 0], poses_obj[2, 0]), scl_ratio)
        ds[0, 0] = ca[0, 0]
        dsum[0, 0] = ds[0, 0]
        imatch[0, 0] = 0  # Match the first target pose to the first object pose
        target_index = [i for i in range(0, n_target)]

        # Fill in left column ...
        for i_t in range(1, n_target):
            ds[i_t, 0] = target_poses[i_t].distance(Pose2D(poses_obj[0, 0], poses_obj[1, 0], poses_obj[2, 0]),
                                                    scl_ratio)
            ca[i_t, 0] = max(ca[i_t - 1, 0], ds[i_t, 0])
            dsum[i_t, 0] = dsum[i_t - 1, 0] + ds[i_t, 0]
            imatch[i_t, 0] = 0  # Match the ith target pose to the first object pose

        # ... and top row
        for i_p in range(1, n_object_path):
            ds[0, i_p] = target_poses[0].distance(Pose2D(poses_obj[0, i_p], poses_obj[1, i_p], poses_obj[2, i_p]),
                                                  scl_ratio)
            ca[0, i_p] = max(ca[0, i_p - 1], ds[0, i_p])
            if ds[0, i_p] < dsum[0, i_p - 1]:
                imatch[0, i_p] = i_p  # Match the first target pose to this object pose
                dsum[0, i_p] = ds[0, i_p]
            else:
                imatch[0, i_p] = imatch[0, i_p - 1]  # Match to an earlier pose
                dsum[0, i_p] = dsum[0, i_p - 1]

        # Remaining matrix
        for i_t in range(1, n_target):
            tp = target_poses[i_t]
            for i_p in range(1, n_object_path):
                ds[i_t, i_p] = tp.distance(Pose2D(poses_obj[0, i_p], poses_obj[1, i_p], poses_obj[2, i_p]), scl_ratio)
                ca[i_t, i_p] = max(min(ca[i_t - 1, i_p],
                                       ca[i_t - 1, i_p - 1],
                                       ca[i_t, i_p - 1]),
                                   ds[i_t, i_p])
                # Compare using this match with the best match upper row so far
                # to best match found so far
                if ds[i_t, i_p] + dsum[i_t - 1, i_p] < dsum[i_t, i_p - 1]:
                    imatch[i_t, i_p] = i_p
                    dsum[i_t, i_p] = ds[i_t, i_p] + dsum[i_t - 1, i_p]
                else:
                    dsum[i_t, i_p] = dsum[i_t, i_p - 1]  # Keep the old match
                    imatch[i_t, i_p] = imatch[i_t, i_p - 1]

        # initialize with minimum value match - allows backtracking
        target_index = []
        v_min = np.amin(ds, axis=1)
        for r in range(0, n_target):
            indx = np.where(ds[r, :] == v_min[r])
            target_index.append(indx[0][0])

        b_is_ok = True
        for i in range(0, n_target - 1):
            if target_index[i + 1] < target_index[i]:
                b_is_ok = False
                print("Frechet: Found array not sorted")

        # Could just do this, but leaving be for a moment to ensure working
        if b_is_ok == False:
            for i_t in range(0, n_target):
                target_index[i_t] = imatch[i_t, n_object_path - 1]

        return ca[n_target - 1, n_object_path - 1], target_index
