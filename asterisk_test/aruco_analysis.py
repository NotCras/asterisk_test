"""
Handles the aruco analysis for a single trial using my custom aruco_tool package.
"""

from aruco_tool import ArucoFunc


class AstArucoAnalysis:

    def __init__(self, file_loc_obj, camera_calib, camera_dists, marker_side_dims):
        pass
        self.af = ArucoFunc(camera_calib, camera_dists, marker_side_dims)

        self.aruco_pics_loc = file_loc_obj.aruco_pics
        self.aruco_data_loc = file_loc_obj.aruco_data

    def aruco_analyze_trial(self, trial_name, aruco_id):
        """
        Analyzes a folder of aruco images
        """
        trial_folder = self.aruco_pics_loc + trial_name #todo:double check that this is correct syntax
        aruco_loc = self.af.full_analysis_single_id(trial_folder, aruco_id)

        result_folder = self.aruco_data_loc + trial_name
        aruco_loc.save_poses(file_name_overwrite=result_folder)

    def load_calibration(self, calibration_loc, dist_loc):
        """
        Function loads camera calibration and dist values from csv files.
        """
        pass




