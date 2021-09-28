from pathlib import Path
from cv2 import aruco
from viz_index_helper import ArucoIndices
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_manager as datamanager
import os

# === IMPORTANT ATTRIBUTES ===
marker_side = 0.03
processing_freq = 1  # analyze every 1 image
# ============================


class ArucoPoseDetect:
    def __init__(self, ar_viz_obj, filter_corners=False, filter_window=3): #, autocrop=False):
        """
        Object for running pose analysis on data
        """
        if filter_corners:
            # makes easier workflow between aruco vision and posedetect objects
            ar_viz_obj.filter_corners(window_size=filter_window)

        self.vision_data = ar_viz_obj
        # camera calibration
        self.mtx = ar_viz_obj.mtx
        # k1,k2,p1,p2 ie radial dist and tangential dist
        self.dist = ar_viz_obj.dist

        self.init_pose, self.est_poses = self.estimate_pose()
        # self.start = 0
        # self.end = None
        #
        # if autocrop:
        #     print("Running autocropper!")
        #     cropper = ArucoAutoCrop(self.est_poses)
        #
        #     start_i, end_i, _, _ = cropper.auto_crop()
        #
        #     print(f"cropped indices => start:{start_i} | end:{end_i}")
        #
        #     self.est_poses = self.est_poses.loc[start_i:end_i]
        #     self.start = start_i
        #     self.end = end_i
    #
    # def get_autocrop_indices(self):
    #     return self.start, self.end

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                example 1) angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                example 2) angle_between((1, 0, 0), (1, 0, 0))
                0.0
                example 3) angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
                *ahem* https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
                (look at highest voted answer, then scroll down to sgt_pepper and crizCraig's answer
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)

        minor = np.linalg.det(
            np.stack((v1_u[-2:], v2_u[-2:]))
        )

        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)

        # # sgt_pepper
        # if minor == 0:
        #     raise NotImplementedError('Too odd vectors =(')
        # return np.sign(minor) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        # # original
        #return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def inverse_perspective(self, rvec, tvec):
        """
        found you! https://aliyasineser.medium.com/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
        """
        # print(rvec)
        # print(np.matrix(rvec[0]).T)
        R, _ = cv2.Rodrigues(rvec)
        R = np.matrix(R).T
        invTvec = np.dot(-R, np.matrix(tvec))
        invRvec, _ = cv2.Rodrigues(R)
        return invRvec, invTvec

    def relative_position(self, rvec1, tvec1, rvec2, tvec2):
        rvec1, tvec1 = np.expand_dims(rvec1.squeeze(),1), np.expand_dims(tvec1.squeeze(),1)
        rvec2, tvec2 = np.expand_dims(rvec2.squeeze(),1), np.expand_dims(tvec2.squeeze(),1)
        invRvec, invTvec = self.inverse_perspective(rvec2, tvec2)

        orgRvec, orgTvec = self.inverse_perspective(invRvec, invTvec)

        info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
        composedRvec, composedTvec = info[0], info[1]

        composedRvec = composedRvec.reshape((3, 1))
        composedTvec = composedTvec.reshape((3, 1))

        return composedRvec, composedTvec

    def estimate_pose(self):
        """
        Estimate the pose of each image using the corners from the aruco vision object
        """
        estimated_poses = pd.DataFrame()

        # get the first set of corners
        ic = self.vision_data.corners.iloc[0]
        _ , init_corners = self.vision_data.row_to_corner(ic)
        init_rvec, init_tvec, _ = aruco.estimatePoseSingleMarkers([init_corners], self.vision_data.marker_side,
                                                                  self.mtx, self.dist)
        init_pose = np.concatenate((init_rvec, init_tvec))
        # orig_corners = orig_corners[0].squeeze()

        total_successes = 0
        final_i = 0

        for i, next_corners in self.vision_data.yield_corners():
            try:
                # print(f"Estimating pose in image {i}")
                next_rvec, next_tvec, _ = aruco.estimatePoseSingleMarkers([next_corners], self.vision_data.marker_side,
                                                                          self.mtx, self.dist)
                # next_corners = next_corners[0].squeeze()

                #print(f"calculating angle, {next_corners}")
                rel_angle = self.angle_between(init_corners[0] - init_corners[2], next_corners[0] - next_corners[2])
                rel_rvec, rel_tvec = self.relative_position(init_rvec, init_tvec, next_rvec, next_tvec)

                translation_val = np.round(np.linalg.norm(rel_tvec), 4)
                rotation_val = rel_angle * 180 / np.pi

                # found the stack overflow for it?
                # https://stackoverflow.com/questions/51270649/aruco-marker-world-coordinates
                rotM = np.zeros(shape=(3, 3))
                cv2.Rodrigues(rel_rvec, rotM, jacobian=0)
                ypr = cv2.RQDecomp3x3(rotM)  # TODO: not sure what we did with this earlier... need to check

                total_successes += 1

            except Exception as e:
                print(f"Error with ARuco corners in image {i}.")
                print(e)
                rel_rvec, rel_tvec = (None, None, None), (None, None, None)
                translation_val = None
                rotation_val = None

            rel_pose = np.concatenate((rel_rvec, rel_tvec))
            rel_pose = rel_pose.squeeze()
            rel_df = pd.Series(
                {"frame": i, "roll": rel_pose[0], "pitch": rel_pose[1], "yaw": rel_pose[2], "x": rel_pose[3],
                 "y": rel_pose[4], "z": rel_pose[5], "tmag": translation_val, "rmag": rotation_val})
            estimated_poses = estimated_poses.append(rel_df, ignore_index=True)
            final_i = i

        # print(" ")
        # print(f"Successfully analyzed: {total_successes} / {final_i+1} corners")
        estimated_poses = estimated_poses.set_index("frame")
        estimated_poses = estimated_poses.round(4)
        return init_pose, estimated_poses

    def save_poses(self, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        :param file_name_overwrite: optional parameter, will save as generate_name unless a different name is specified
        """
        if file_name_overwrite is None:
            data_name = self.vision_data.trial_name
            folder = "aruco_data"  # "csv"
            new_file_name = f"{folder}/{data_name}.csv"

        else:
            new_file_name = file_name_overwrite + ".csv"

        self.est_poses.to_csv(new_file_name, index=True)
        # print(f"CSV File generated with name: {new_file_name}")

    def plot_est_pose(self):
        """
        Plots translation data (x, y)
        """
        # pdb.set_trace()
        self.est_poses.plot(x="x", y="y")

if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()