#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kartik (original), john (major edits, cleaning)
"""
import numpy as np
import sys
import cv2
from cv2 import aruco
from pathlib import Path
import os
import time
import pandas as pd
import asterisk_data_manager as datamanager

# === IMPORTANT ATTRIBUTES ===
marker_side = 0.03
processing_freq = 1  # analyze every 1 image
# ============================


class AsteriskArucoVision:

    def __init__(self, side_dims=0.03, freq=1):
        """
        Object for running aruco vision analysis on a trial of images
        """
        self.marker_side = side_dims
        self.processing_freq = freq
        self.home_directory = Path(__file__).parent.absolute()

        self.mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                        (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                        (0, 0, 1)))
        # k1,k2,p1,p2 ie radial dist and tangential dist
        self.dist = np.array((0.1611730644, -0.3392379107, 0.0010744837,	0.000905697))

        self.origin = None
        self.path = None

    def inverse_perspective(self, rvec, tvec):
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

    def estimate_pose(self, frame, marker_side, mtx, dist):

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250) # MAKE SURE YOU HAVE RIGHT ONE!!!!
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        # parameters.adaptiveThreshConstant = 10

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # print(f"corners: {corners}, ids: {ids}")

        if np.all(ids != None):
            # print("Found a tag.")
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_side, mtx, dist)
            # TODO: quickfix is corners[0]... is there a more elegant way to fix?

        else:
            print("Could not find marker in frame.")
            rvec, tvec = (None, None, None), (None, None, None)
            # TODO: fix reference before assignment here
            # quit()

        return rvec, tvec, corners

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
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def pose_estimation_process(self, folder, image_tag, mtx_val, dist_val, init_corners, init_rvec, init_tvec):
        frame = cv2.imread(os.path.join(folder, image_tag))

        next_rvec, next_tvec, next_corners = self.estimate_pose(
            frame, marker_side, mtx_val, dist_val)
        next_corners = next_corners[0].squeeze()

        # print(f"calculating angle, {next_corners}")
        rel_angle = self.angle_between(
            init_corners[0]-init_corners[2], next_corners[0]-next_corners[2])
        rel_rvec, rel_tvec = self.relative_position(
            init_rvec, init_tvec, next_rvec, next_tvec)

        translation_val = np.round(np.linalg.norm(rel_tvec),4)
        rotation_val = rel_angle*180/np.pi

        rotM = np.zeros(shape=(3, 3))
        cv2.Rodrigues(rel_rvec, rotM, jacobian=0)
        ypr = cv2.RQDecomp3x3(rotM)

        return rel_rvec, rel_tvec, translation_val, rotation_val, ypr

    # mtx = camera intrinsic matrix , dist =  distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])

    # ================================================================
    def analyze_images(self, data_path, subject_name, hand_name, t_label, r_label, trial_number):
        # make empty dataframe
        estimated_poses = pd.DataFrame()

        frame = cv2.imread(os.path.join(data_path, 'left0000.jpg'))
        orig_rvec, orig_tvec, orig_corners = self.estimate_pose(
            frame, marker_side, self.mtx, self.dist)
        orig_corners = orig_corners[0].squeeze()
        # print("Tag found in initial image.")

        # put original value in dataframe as first index
        orig_pose = np.concatenate((orig_rvec, orig_tvec))
        orig_df = pd.Series({"roll": orig_pose[0], "pitch": orig_pose[1], "yaw":orig_pose[2], "x":orig_pose[3],
                             "y": orig_pose[4], "z": orig_pose[5], "tmag": 0, "rmag": 0})
        estimated_poses = estimated_poses.append(orig_df)

        analyzed_successfully = 0
        total_counter = 0

        f = []

        for (dirpath, dirnames, filenames) in os.walk(data_path):
            f.extend(filenames)
            f.sort()
            break

        # print statements for debugging
        # print(f)
        # print(" ")
        # print(dirpath)
        # print(" ")

        data_file = f"{subject_name}_{hand_name}_{t_label}_{r_label}_{trial_number}.csv"
        csv_loc = f"csv/{data_file}"

        while True:
            for image_ in f:
                # print(image_)
                if '.ini' in image_:
                    # print("Configuration file found. Skipping over.")
                    # camera configuration file, skip over
                    continue

                if np.mod(total_counter, processing_freq) > 0:
                    continue

                try:
                    rel_rvec, rel_tvec, translation, rotation, ypr = self.pose_estimation_process(
                        data_path, image_, self.mtx, self.dist, orig_corners, orig_rvec, orig_tvec)
                    # print(f"Succeeded at image {counter}")
                    
                except Exception as e: 
                    print(f"Error with finding ARuco tag in image {total_counter}.")
                    print(e)
                    total_counter += 1
                    continue

                total_counter += 1
                analyzed_successfully += 1

                rel_pose = np.concatenate((rel_rvec, rel_tvec))

                rel_df = pd.Series(
                    {"roll": rel_pose[0], "pitch": rel_pose[1], "yaw": rel_pose[2], "x": rel_pose[3],
                     "y": rel_pose[4], "z": rel_pose[5], "tmag": translation, "rmag": rotation})
                estimated_poses = estimated_poses.append(rel_df)

                # with open(csv_loc,'a') as fd:
                #     for i in rel_pose:
                #         # for y in i:
                #         fd.write(str(i[0]))
                #         fd.write(',')
                #         # print('here')
                #
                #     fd.write(str(translation))
                #     fd.write(',')
                #     fd.write(str(rotation))
                #     fd.write('\n')
    #                    print(rel_pose)

                # print('Total: ' + str(total) +' Done '+ image_)

            self.save_data(estimated_poses, csv_loc)
            print('Completed ' + data_file)
            print("Finished: " + str(analyzed_successfully) + "/" + str(total_counter))
            print("          ")
            break

    def save_data(self, data_df, location):
        """
        Saves most recent data
        """
        data_df.to_csv(location)


if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    vision = AsteriskArucoVision()  # using defaults

    print("""
            ========= ASTERISK TEST ARUCO ANALYSIS ==========
            I ANALYZE YOUR VISION DATA FOR THE ASTERISK STUDY
                  AT NO COST, STRAIGHT TO YOUR DOOR!
                               *****

            What can I help you with?
            1 - view a set of images like a video
            2 - aruco analyze one specific set of data
            3 - aruco analyze a batch of data
            4 - validate aruco pose on images
        """)

    ans = datamanager.smart_input("Enter a function", "mode", ["1", "2", "3"])
    subject = datamanager.smart_input("Enter subject name: ", "subjects")
    hand = datamanager.smart_input("Enter name of hand: ", "hands")

    if ans == "1":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotations")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        viewer = datamanager.AstData()
        viewer.view_images(subject, hand, translation, rotation, trial_num)

    elif ans == "2":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotations")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
        folder_path = f"viz/{file_name}/"

        try:
            vision.analyze_images(folder_path, subject, hand, translation, rotation, trial_num)
        except Exception as e:
            print(e)

        print(f"Completed Aruco Analysis for: {file_name}")

    elif ans == "3":
        files_covered = list()

        for s, h, t, r, n in datamanager.generate_names_with_s_h(subject, hand):
            file_name = f"{s}_{h}_{t}_{r}_{n}"

            folder_path = f"viz/{file_name}/"
            os.chdir(home_directory)
            # data_path = inner_path
            print(folder_path)

            try:
                vision.analyze_images(folder_path, s, h, t, r, n)
                files_covered.append(file_name)
            except Exception as e:
                print(e)
                files_covered.append(f"FAILED: {file_name}")

        print("Completed Batch Aruco Analysis!")
        print(files_covered)

    elif ans == "4":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotations")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")




        pass
