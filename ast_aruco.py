#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four classes for asterisk test aruco analysis:
1) ArucoIndices: finds start and end
2) ArucoVision: handles detecting of aruco code, can filter aruco corners
3) ArucoPoseDetect: given an arucovision object (with detected aruco code corners), calculates aruco code pose
4) ArucoAutoCrop: automatically finds start and end indices using a simple model
5) 2 helper functions which handle aruco code analysis
----- one for single trial analysis, and another for batches of trials
@author: kartik (original), john (major edits, cleaning/refactoring, indices/autocropper)
"""
import numpy as np
import sys, os, time, pdb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import data_manager as datamanager
from cv2 import aruco
from pathlib import Path
from math import isclose
from matplotlib.widgets import Slider, Button
from viz_index_helper import ArucoIndices
from viz_aruco_corners import ArucoVision
from viz_aruco_pose import ArucoPoseDetect
from viz_autocrop import ArucoAutoCrop

# TODO: what should I do with these functions? Make an AstAruco class?
class AstAruco:
    @staticmethod
    def single_aruco_analysis(subject, hand, translation, rotation, trial_num, home=None, indices=True, crop=True):
        # TODO: add considerations of home folder
        file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
        folder_path = f"{file_name}/"

        if indices:
            try:
                b_idx, e_idx = ArucoIndices.get_indices(file_name)
                needs_cropping = False
            except:
                print(f"Failed to get cropped indices for {file_name}")
                e_idx = None
                b_idx = 0
                needs_cropping = True

        else:  # TODO: make more straightforward later
            e_idx = None
            b_idx = 0
            needs_cropping = True

        if not crop:
            needs_cropping = False

        try:
            # print(f"Needs cropping: {needs_cropping}")
            trial = ArucoVision(file_name, begin_idx=b_idx, end_idx=e_idx)
            trial_pose = ArucoPoseDetect(trial, filter_corners=True, filter_window=4)
            print(f"Completed Aruco Analysis for: {file_name}")

        except Exception as e:
            print(e)
            print(f"Failed Aruco Analysis for: {file_name}")  # TODO: be more descriptive about where the error happened
            return

        try:
            if needs_cropping:
                if r in ['cw', 'ccw']:
                    trial_cropped = ArucoAutoCrop(trial_pose, only_rotation=True)
                else:
                    trial_cropped = ArucoAutoCrop(trial_pose)

                trial_cropped.save_poses()
            else:
                trial_pose.save_poses()

        except Exception as e:
            print(e)
            print(f"Failed ArucoAutoCrop for: {file_name}")

    @staticmethod
    def batch_aruco_analysis(subject, hand, no_rotations=True, home=None, indices=True, crop=True):
        files_covered = list()
        files_df = pd.DataFrame(columns=["name", "start_i", "end_i"])

        for s, h, t, r, n in datamanager.generate_names_with_s_h(subject, hand, no_rotations=no_rotations):
            file_name = f"{s}_{h}_{t}_{r}_{n}"

            folder_path = f"{file_name}/"
            if home is not None:
                os.chdir(home)
            # data_path = inner_path
            print(folder_path)

            if indices:
                try:
                    b_idx, e_idx = ArucoIndices.get_indices(file_name)
                    needs_cropping = False
                except:
                    e_idx = None
                    b_idx = 0
                    needs_cropping = True

            else:  # TODO: make more straightforward later
                e_idx = None
                b_idx = 0
                needs_cropping = True

            if not crop:
                needs_cropping = False

            try:
                trial = ArucoVision(file_name, begin_idx=b_idx, end_idx=e_idx)
                trial_pose = ArucoPoseDetect(trial, filter_corners=True, filter_window=4)

            except Exception as e:
                print(e)
                files_covered.append(f"FAILED: {file_name}")
                files_df = files_df.append({"name": file_name, "start_i": -1, "end_i": -1}, ignore_index=True)
                continue

            try:
                if needs_cropping:
                    print("Running Autocropper!")

                    if r in ['cw', 'ccw']:
                        trial_cropped = ArucoAutoCrop(trial_pose, only_rotation=True)
                    else:
                        trial_cropped = ArucoAutoCrop(trial_pose)

                    trial_cropped.save_poses()
                    s, e = trial_cropped.get_autocrop_indices()
                    files_df = files_df.append({"name": file_name, "start_i": s, "end_i": e}, ignore_index=True)

                else:
                    trial_pose.save_poses()

                files_covered.append(file_name)

            except Exception as e:
                print(e)
                files_covered.append(f"FAILED indices: {file_name}")
                files_df = files_df.append({"name": file_name, "start_i": -1, "end_i": -1}, ignore_index=True)

            files_df.to_csv(f"index_values_{hand}.csv")

        print("Completed Batch Aruco Analysis!")
        print(files_covered)


if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    print("""
            ========= ASTERISK TEST ARUCO ANALYSIS ==========
            I ANALYZE YOUR VISION DATA FOR THE ASTERISK STUDY
                  AT NO COST, STRAIGHT TO YOUR DOOR!
                               *****

            What can I help you with?
            1 - view a set of images like a video
            2 - aruco analyze one specific set of data
            3 - aruco analyze a batch of data
            4 - validate aruco corner detection on images in slow-mo video
            5 - test whether the get_index function will get your indices correctly
            6 - use a helper to find index values for a certain trial
        """)

    ans = datamanager.smart_input("Enter a function", "mode", ["1", "2", "3", "4", "5", "6"])
    subject = datamanager.smart_input("Enter subject name: ", "subjects")
    hand = datamanager.smart_input("Enter name of hand: ", "hands")

    if ans == "1":
        translation = datamanager.smart_input("Enter type of translation: ", "translations_w_n")

        if translation == "n":
            rotation = datamanager.smart_input("Enter type of rotation: ", "rotations_n_trans")
        else:
            rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")

        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        viewer = datamanager.AstData()
        viewer.view_images(subject, hand, translation, rotation, trial_num)

    elif ans == "2":
        translation = datamanager.smart_input("Enter type of translation: ", "translations_w_n")

        if translation == "n":
            rotation = datamanager.smart_input("Enter type of rotation: ", "rotations_n_trans")
        else:
            rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")

        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")
        index = datamanager.smart_input("Should we search for stored index values (start & end)", "consent")
        crop = datamanager.smart_input("Should we try to automatically crop the trial's start and end?", "consent")

        i = index == 'y'
        c = crop == 'y'  # TODO: work on reducing number of prompts?

        single_aruco_analysis(subject, hand, translation, rotation, trial_num, home=home_directory, indices=i, crop=c)

    elif ans == "3":
        r = datamanager.smart_input("Should we include trials with rotations? ", "consent")
        index = datamanager.smart_input("Should we search for stored index values (start & end)", "consent")
        crop = datamanager.smart_input("Should we try to automatically crop the trial's start and end?", "consent")

        rots = r == 'n'
        i = index == 'y'
        c = crop == 'y'

        batch_aruco_analysis(subject, hand, no_rotations=rots, home=home_directory, indices=i, crop=c)

    elif ans == "4":
        pass

    elif ans == "5":
        pass
    elif ans == "6":
        pass



