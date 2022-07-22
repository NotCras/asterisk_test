
from pathlib import Path
from cv2 import aruco
from viz_index_helper import ArucoIndices
from viz_autocrop import ArucoAutoCrop
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

    def get_trial_name(self):
        """
        Get trial name.
        """
        return self.trial_name

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

if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    subject = datamanager.smart_input("Enter subject name: ", "subjects")
    hand = datamanager.smart_input("Enter name of hand: ", "hands")
    translation = datamanager.smart_input("Enter type of translation: ", "translations_w_n")

    if translation == "n":
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotations_n_trans")
    else:
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")

    trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

    file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"
    folder_path = f"{file_name}/"

    index = datamanager.smart_input("Should we search for stored index values (start & end)", "consent")

    i = index == 'y'

    if i:
        try:
            b_idx, e_idx = ArucoIndices.get_indices(file_name)
        except:
            e_idx = None
            b_idx = 0
            c = True

    else:  # TODO: make more straightforward later
        e_idx = None
        b_idx = 0

    trial = ArucoVision(folder_path, begin_idx=b_idx, end_idx=e_idx)

    trial.filter_corners(window_size=4)  # window size 4 might be better? Very small lag
    trial.validate_corners()

    # extra debugging stuff
    # trial_pose = ArucoPoseDetect(trial, filter_corners=True, filter_window=4)
    # print(f"Missing: {trial_pose.est_poses['x'].isna().sum()}")
    # trial_pose.plot_est_pose()
    # plt.show()