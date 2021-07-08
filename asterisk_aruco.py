#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kartik (original), john (major edits, cleaning)
"""
import numpy as np
import sys, os, time, pdb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import asterisk_data_manager as datamanager
from cv2 import aruco
from pathlib import Path
from math import isclose
from matplotlib.widgets import Slider, Button


# === IMPORTANT ATTRIBUTES ===
marker_side = 0.03
processing_freq = 1  # analyze every 1 image
# ============================

class ArucoIndices:
    """
    Handles data indices, beginning and ending indices
    """

    @staticmethod
    def get_indices(id, file_name=None):
        """
        Gets the beginning and ending indices of the
        :param id:
        :param file_name:
        :return:
        """
        if file_name is not None:
            table = pd.read_csv(file_name)
        else:
            # table = pd.read_csv("viz_data_indices.csv")
            table = pd.read_csv("viz_data_indices_main.csv")

        table = table.set_index("id")
        try:
            indices = table.loc[id]

        except Exception as e:
            print("Could not find the index.")
            print(e)
            raise IndexError("Could not find the correct indices")

        return int(indices["begin_idx"]), int(indices["end_idx"])

    @staticmethod
    def find_indices(folder_path):
        """
        Helper to get the data indices for the start and end of a trial
        """
        home = Path(__file__).parent.absolute()
        os.chdir(folder_path)
        files = [f for f in os.listdir('.') if f[-3:] == 'jpg']
        files.sort()

        # get start index
        start = ArucoIndices._slider_window(files, title="START")

        # get end index
        end = ArucoIndices._slider_window(files, title="END", init_val=int(start+1))

        print(folder_path)
        print(f"start: {start}, end: {end}")

        os.chdir(home)

        if end <= start:
            raise ValueError("starting index cannot be greater than ending index!")

        return start, end

        # TODO: add saving functionality, add these values to the stored index values?

    @staticmethod
    def _slider_window(list_of_files, title, init_val=0):
        """
        Generates a matplotlib window with a slider that changes which image is shown
        """
        num_files = len(list_of_files)

        # setup
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # show initial image
        image_id = list_of_files[0]
        image = plt.imread(image_id)
        plt.imshow(image)
        plt.draw()
        plt.title(f"Get {title} index!")

        # setup slider
        slider_bkd_color = "lightgoldenrodyellow"
        axpos = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=slider_bkd_color)
        # allowed_positions = np.linspace(0, 1, num_files).tolist()
        image_pos = Slider(
            ax=axpos,
            label="Image Number",
            valmin=0,
            valmax=num_files-1,
            valinit=init_val,
            valstep=1
            #init_color="none"
        )

        def _slider_update(val):
            print(f"val: {val}")
            image_id = list_of_files[int(val)]
            image = plt.imread(image_id)
            ax.imshow(image)
            ax.draw()  # if I use plt.draw() it will draw the image in the button's space

        image_pos.on_changed(_slider_update)

        ax_exit = plt.axes([0.8, 0.025, 0.13, 0.05])
        ax_left = plt.axes([0.3, 0.025, 0.1, 0.05])
        ax_right = plt.axes([0.5, 0.025, 0.1, 0.05])

        button_exit = Button(ax_exit, 'Got Index!', color=slider_bkd_color, hovercolor='0.975')
        button_left = Button(ax_left, '<-', color=slider_bkd_color, hovercolor='0.975')
        button_right = Button(ax_right, '->', color=slider_bkd_color, hovercolor='0.975')

        def _button_exit(val):
            print(f"start index: {image_pos.val}")
            plt.close()

        def _button_left(val):
            i = image_pos.val
            if i <= 0:
                i = 0
            else:
                i = i - 1

            image_pos.set_val(i)

        def _button_right(val):
            i = image_pos.val
            if i >= num_files-1:
                i = num_files-1
            else:
                i = i + 1

            image_pos.set_val(i)

        button_exit.on_clicked(_button_exit)
        button_left.on_clicked(_button_left)
        button_right.on_clicked(_button_right)

        plt.show()
        return image_pos.val


class ArucoVision:
    def __init__(self, trial_name, side_dims=0.03, freq=1, begin_idx=0, end_idx=None):
        """
        """
        self.home = Path(__file__).parent.absolute()
        self.trial_name = trial_name
        self.folder_name = f"{trial_name}/"
        self.data_folder = self.home / "viz" / self.folder_name
        self.marker_side = side_dims
        self.marker_side = side_dims
        self.processing_freq = freq

        # camera calibration
        self.mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                             (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                             (0, 0, 1)))
        # k1,k2,p1,p2 ie radial dist and tangential dist
        self.dist = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))

        os.chdir(self.data_folder)
        self.corners = self.analyze_images(end_idx=end_idx, begin_idx=begin_idx)
        os.chdir(self.home)

    def corner_to_series(self, i, corn):
        """
        Convert standard numpy array of corners into pd.Series (to add to corners dataframe)
        """
        c1 = corn[0]
        c2 = corn[1]
        c3 = corn[2]
        c4 = corn[3]
        corner_series = pd.Series({"frame": i, "c1_x": c1[0], "c1_y": c1[1],
                                 "c2_x": c2[0], "c2_y": c2[1],
                                 "c3_x": c3[0], "c3_y": c3[1],
                                 "c4_x": c4[0], "c4_y": c4[1]})
        return corner_series

    def row_to_corner(self, corn):
        """
        Convert one row of dataframe into standard corner numpy array
        """
        i = corn.name
        #pdb.set_trace()
        c1 = [corn["c1_x"], corn["c1_y"]]
        c2 = [corn["c2_x"], corn["c2_y"]]
        c3 = [corn["c3_x"], corn["c3_y"]]
        c4 = [corn["c4_x"], corn["c4_y"]]

        return int(i), np.array([c1, c2, c3, c4], dtype=np.dtype("float32"))

    def _moving_average(self, window_size=3):
        """
        Runs a moving average on the corner data. Saves moving average data into new columns with f_ prefix.
        Overwrites previous moving average calculations.
        :param window_size: size of moving average. Defaults to 3.
        """
        # TODO: makes a bunch of nan values at end of data
        filtered_df = pd.DataFrame()
        #pdb.set_trace()
        filtered_df["frame"] = self.corners.index

        filtered_df["c1_x"] = self.corners["c1_x"].rolling(
            window=window_size, min_periods=1).mean()
        filtered_df["c1_y"] = self.corners["c1_y"].rolling(
            window=window_size, min_periods=1).mean()

        filtered_df["c2_x"] = self.corners["c2_x"].rolling(
            window=window_size, min_periods=1).mean()
        filtered_df["c2_y"] = self.corners["c2_y"].rolling(
            window=window_size, min_periods=1).mean()

        filtered_df["c3_x"] = self.corners["c3_x"].rolling(
            window=window_size, min_periods=1).mean()
        filtered_df["c3_y"] = self.corners["c3_y"].rolling(
            window=window_size, min_periods=1).mean()

        filtered_df["c4_x"] = self.corners["c4_x"].rolling(
            window=window_size, min_periods=1).mean()
        filtered_df["c4_y"] = self.corners["c4_y"].rolling(
            window=window_size, min_periods=1).mean()

        filtered_df = filtered_df.round(4)
        filtered_df = filtered_df.set_index("frame")
        return filtered_df

    def filter_corners(self, window_size=3):
        """
        Overwrite the corner data with filtered version
        """
        self.corners = self._moving_average(window_size)

    def save_corners(self, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        :param file_name_overwrite: optional parameter, will save as generate_name unless a different name is specified
        """
        if file_name_overwrite is None:
            file_name = self.data_folder.stem
            file_name = file_name.replace("/", "_")
            new_file_name = file_name + ".csv"

        else:
            new_file_name = file_name_overwrite + ".csv"

        self.corners.to_csv(new_file_name, index=True)
        # print(f"CSV File generated with name: {new_file_name}")

    def yield_corners(self):
        """
        yields corners as numpy array
        """
        # pdb.set_trace()
        for i, row in self.corners.iterrows():
            yield self.row_to_corner(row)

    def get_images(self, idx_limit=None, idx_bot=0):
        """
        Retrieve list of image names, sorted.
        NOTE: Need to manually change directory back to self.home
        """
        # TODO: be smarter about indices. If bot != 0, then bot-1 | if idx_limit != len(files), then lim + 1
        os.chdir(self.data_folder)
        files = [f for f in os.listdir('.') if f[-3:] == 'jpg']
        files.sort()

        #print(f"Num of image files in folder: {len(files)}")
        if idx_limit is not None:
            if idx_bot != 0:
                idx_bot = idx_bot-1  # this is so that we can include idx_bot

            if idx_limit != len(files):  # so that we can include idx_limit
                idx_limit = idx_limit + 1

            try:
                files = files[idx_bot:idx_limit]

            except Exception as e:
                print("get_images error: ")
                print(e)

        return files

    def analyze_images(self, begin_idx=0, end_idx=None):
        corner_data = pd.DataFrame()
        files = self.get_images(idx_limit=end_idx, idx_bot=begin_idx)

        # set up aruco dict and parameters
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        aruco_params = aruco.DetectorParameters_create()

        for i, f in enumerate(files):
            image = cv2.imread(f)

            # make image black and white
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # get estimated aruco pose
            corners, ids, _ = aruco.detectMarkers(image=image_gray, dictionary=aruco_dict,
                                                  parameters=aruco_params, cameraMatrix=self.mtx,
                                                  distCoeff=self.dist)

            # print(ids)
            try:
                if 2 not in ids:
                    print(ids)
                    print("FOUND WRONG ARUCO CODE, CANT FIND CORRECT ONE")
                    raise ValueError("Did not find correct aruco code.")

                if len(ids) > 1:
                    # TODO: so this works, but maybe add some better tracking of this by considering ids and their index
                    print(f"More than one aruco tag found at frame {i}!")
                    corners = corners[0]

                c = corners[0].squeeze()
            except Exception as e:
                print(f"Failed to find an aruco code at frame {i}!")
                print(e)
                # make corners of None to make sure that we log the failed attempt to find aruco code
                c = np.array([[None, None], [None, None], [None, None], [None, None]])
                # pdb.set_trace()

            # pdb.set_trace()
            corner_series = self.corner_to_series(i, c)
            corner_data = corner_data.append(corner_series, ignore_index=True)

        corner_data = corner_data.set_index("frame")

        os.chdir(self.home)
        return corner_data

    def plot_corners(self, frame_num):
        colors = ["r", "b", "g", "gray"]

        point = self.corners.loc[frame_num]

        # plot image with point plotted on top
        for i in range(4):
            x = point[f"c{i+1}_x"]
            y = point[f"c{i+1}_y"]

            plt.plot(x, y, marker="o", color=colors[i], fillstyle='none')

    def validate_corners(self, delay=0.1, take_input=True):
        """
        Draws corners on image for visual debugging
        """
        # TODO: add more onto the plot: title, data numbers?, center?, frame number?
        files = self.get_images()
        for i, f in enumerate(files):
            plt.clf()
            image = plt.imread(f)
            plt.imshow(image)
            self.plot_corners(i)

            plt.pause(delay)
            plt.draw()

        os.chdir(self.home)
        if take_input:
            repeat = datamanager.smart_input("Show again? [y/n]", "consent")
            if repeat == "y":
                # run again
                self.validate_corners(delay, take_input)
            else:
                # stop running
                quit()


class ArucoPoseDetect:
    def __init__(self, ar_viz_obj, filter_corners=False, filter_window=3, autocrop=False):
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

        # TODO: add option for autocropper

        self.init_pose, self.est_poses = self.estimate_pose()

        if autocrop:
            cropper = ArucoAutoCrop(self.est_poses)

            start_i, end_i, _, _ = cropper.auto_crop()

            self.est_poses = self.est_poses.loc[start_i:end_i]



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
                rel_angle = self.angle_between(
                    init_corners[0] - init_corners[2], next_corners[0] - next_corners[2])
                rel_rvec, rel_tvec = self.relative_position(
                    init_rvec, init_tvec, next_rvec, next_tvec)

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


class ArucoAutoCrop:
    def __init__(self, df_data):
        self.trial_data = df_data

    def yield_index_pairs(self, desired_rotation=None):
        """
        yields pairs of indices to check auto cropping on
        first index goes through all of the indices except the last index in the data
        the second index goes through all of the indices between the first index and the last index
        :return:
        """
        data_size = len(self.trial_data)

        if desired_rotation is not None and (isinstance(desired_rotation, int) or isinstance(desired_rotation, float)):
            # find the first index that achieves the desired rotation in the data set
            # self.trial_data.rmag.eq(15).idxmax()
            i = 0
            for val in self.trial_data.rmag:
                if abs(val-15) < 0.5:  # TODO: revisit this logic
                    start_i = i
                    break

                i += 1
                pass

        else:
            start_i = 0

        for i1 in range(start_i, data_size - 1):
            for i2 in range(i1 + 1, data_size):
                yield i1, i2

    def auto_crop(self):
        """
        Crops an image trial automatically, but finding the largest distance travelled in the smallest range of index
        :param df_data
        :return:
        """
        trial_length = len(self.trial_data)
        c_max_is = (1, trial_length)

        c_max_dist = 0
        c_min_di = trial_length

        for i1, i2 in self.yield_index_pairs():
            d1 = self.trial_data.iloc[i1]  # TODO: check function call
            d2 = self.trial_data.iloc[i2]

            i_dist = np.sqrt((d1['x']+d2['x'])**2 + (d1['y']+d2['y'])**2)  # the distance between the sampled points
            d_i = i2 - i1

            # now check for...
            if c_max_dist <= i_dist:  # if we record a greater distance than before...

                if c_min_di >= d_i:  # is the space between the indices smaller than before?
                    c_max_dist = i_dist
                    c_min_di = d_i

                    c_max_is = (i1, i2)  # record which indices we are saving

        return c_max_is[1], c_max_is[2], c_max_dist, c_min_di


def single_aruco_analysis(subject, hand, translation, rotation, trial_num, home=None, indices=True, crop=True):
    # TODO: add considerations of home folder
    file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
    folder_path = f"{file_name}/"


    try:
        b_idx, e_idx = ArucoIndices.get_indices(file_name)
        needs_cropping = False
    except:
        print(f"Failed to get cropped indices for {file_name}")
        e_idx = None
        b_idx = 0
        needs_cropping = True

    if not indices:  # TODO: make more straightforward later
        e_idx = None
        b_idx = 0

    if not crop:
        needs_cropping = False

    try:
        trial = ArucoVision(file_name, begin_idx=b_idx, end_idx=e_idx)
        trial_pose = ArucoPoseDetect(trial, filter_corners=True, filter_window=4, autocrop=needs_cropping)

        trial_pose.save_poses()
        print(f"Completed Aruco Analysis for: {file_name}")

    except Exception as e:
        print(e)
        print(f"Failed Aruco Analysis for: {file_name}")  # TODO: be more descriptive about where the error happened


def batch_aruco_analysis(subject, hand, no_rotations=True, home=None, indices=True, crop=True):
    files_covered = list()
    for s, h, t, r, n in datamanager.generate_names_with_s_h(subject, hand, no_rotations=no_rotations):
        file_name = f"{s}_{h}_{t}_{r}_{n}"

        folder_path = f"{file_name}/"
        if home is not None:
            os.chdir(home)
        # data_path = inner_path
        print(folder_path)

        try:
            b_idx, e_idx = ArucoIndices.get_indices(file_name)
            needs_cropping = False
        except:
            e_idx = None
            b_idx = 0
            needs_cropping = True

        if not indices:  # TODO: make more straightforward later
            e_idx = None
            b_idx = 0

        if not crop:
            needs_cropping = False

        try:
            trial = ArucoVision(file_name, begin_idx=b_idx, end_idx=e_idx)
            trial_pose = ArucoPoseDetect(trial, filter_corners=True, filter_window=4, autocrop=needs_cropping)
            trial_pose.save_poses()

            files_covered.append(file_name)
        except Exception as e:
            print(e)
            files_covered.append(f"FAILED: {file_name}")

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
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        viewer = datamanager.AstData()
        viewer.view_images(subject, hand, translation, rotation, trial_num)

    elif ans == "2":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")
        index = datamanager.smart_input("Should we search for stored index values (start & end)", "consent")
        crop = datamanager.smart_input("Should we try to automatically crop the trial's start and end?", "consent")

        i = index == 'y'
        c = crop == 'y' # TODO: work on reducing number of prompts?

        single_aruco_analysis(subject, hand, translation, rotation, trial_num, home=home_directory, indices=i, crop=c)

    elif ans == "3":
        index = datamanager.smart_input("Should we search for stored index values (start & end)", "consent")
        crop = datamanager.smart_input("Should we try to automatically crop the trial's start and end?", "consent")

        i = index == 'y'
        c = crop == 'y'

        batch_aruco_analysis(subject, hand, no_rotations=True, home=home_directory, indices=i, crop=c)

    elif ans == "4":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
        folder_path = f"{file_name}/"

        b_idx, e_idx = ArucoIndices.get_indices(file_name)
        trial = ArucoVision(folder_path, begin_idx=b_idx, end_idx=e_idx)

        trial.filter_corners(window_size=4)  # window size 4 might be better? Very small lag
        trial.validate_corners()

        # extra debugging stuff
        # trial_pose = ArucoPoseDetect(trial, filter_corners=True, filter_window=4)
        # print(f"Missing: {trial_pose.est_poses['x'].isna().sum()}")
        # trial_pose.plot_est_pose()
        # plt.show()

    elif ans == "5":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
        folder_path = f"{file_name}/"

        try:
            b_idx, e_idx = ArucoIndices.get_indices(file_name)
        except:
            e_idx = None
            b_idx = 0

        print(f"b: {b_idx}, e: {e_idx}")

    elif ans == "6":
        translation = datamanager.smart_input("Enter type of translation: ", "translations")
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")
        trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

        file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
        folder_path = f"viz/{file_name}/"

        s_i, e_i = ArucoIndices.find_indices(folder_path)


