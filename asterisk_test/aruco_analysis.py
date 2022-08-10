"""
Handles the aruco analysis for a single trial using my custom aruco_tool package.
"""

from aruco_tool import ArucoFunc
import data_manager as datamanager

import logging as log


class AstArucoAnalysis:
    # dict which contains the mapping between aruco ids and hand
    aruco_hand_to_id = {}

    def __init__(self, file_loc_obj, camera_calib, camera_dists, marker_side_dims):
        pass
        self.af = ArucoFunc(camera_calib, camera_dists, marker_side_dims)

        self.aruco_pics_loc = file_loc_obj.aruco_pics
        self.aruco_data_loc = file_loc_obj.aruco_data

    def aruco_analyze_trial(self, trial_name, aruco_id, save_trial=False):
        """
        Analyzes a folder of aruco images
        """
        h, t, r, s, n = trial_name.split("_")
        trial_attributes = {"hand":h, "translation":t, "rotation":r, "subject":s, "trial_num":n}

        trial_folder = self.aruco_pics_loc / f"{s}_{h}_{t}_{r}_{n}" # TODO:double check that this is correct syntax
        aruco_loc = self.af.full_analysis_single_id(trial_folder, aruco_id)

        aruco_loc.data_attributes = trial_attributes

        if save_trial:
            result_folder = self.aruco_data_loc / trial_name
            aruco_loc.save_poses(file_name_overwrite=result_folder)

        return aruco_loc

    def load_calibration(self, calibration_loc, dist_loc):
        """
        Function loads camera calibration and dist values from csv files.
        """
        pass

    def batch_aruco_analysis(self, subject, hand, exclude_rotations=True, save_data=True,
                             assess_indices=False, crop_trial=False):
        """
        Runs aruco analysis on a set of trials. Collects data into a dictionary (trial_name) -> (trial data, as aruco_loc obj)
        """
        files_covered = list()
        batch_trial_data = dict()

        for s, h, t, r, n in datamanager.generate_names_with_s_h(subject, hand, no_rotations=exclude_rotations):
            trial_name = f"{s}_{h}_{t}_{r}_{n}" # TODO: add generate filename to datamanager to handle this
            log.info(f"Attempting {trial_name}, aruco analysis.")

            #TODO: use try, except here
            trial_data = self.aruco_analyze_trial(trial_name, self.aruco_hand_to_id[hand], save_trial=save_data)

            if assess_indices:
                # find indices to crop to in the data
                # TODO: do this
                raise NotImplementedError("Can't assess indices yet.")

                if crop_trial: #and follow_through_with_crop:
                    # actually crop the trial
                    # TODO: do this
                    raise NotImplementedError("Can't crop trial data by indices yet.")

            batch_trial_data[trial_name] = trial_data
            files_covered.append(trial_name)

            log.info(f"Succeeded: {trial_name}, aruco analysis.")

        log.info(f"========  Batch aruco analysis complete for {hand}, {subject}.")
        return batch_trial_data

# TODO: add the script portion, just like ast_aruco







