"""
Handles the aruco analysis for a single trial using my custom aruco_tool package.
"""

from aruco_tool import ArucoFunc
import data_manager as datamanager
from file_manager import my_ast_files
from ast_hand_info import get_hand_stats


import logging as log
from pathlib import Path
import numpy as np
import pdb


class AstArucoAnalysis:
    # dict which contains the mapping between aruco ids and hand
    aruco_hand_to_id = {}

    def __init__(self, file_loc_obj, camera_calib, camera_dists, marker_side_dims):
        self.af = ArucoFunc(camera_calib, camera_dists, marker_side_dims)

        self.aruco_pics_loc = file_loc_obj.aruco_pics
        self.aruco_data_loc = file_loc_obj.aruco_data

    def aruco_analyze_trial(self, trial_name, aruco_id, save_trial=False):
        """
        Analyzes a folder of aruco images
        """
        h, t, r, s, n = trial_name.split("_")
        #trial_attributes = {"hand": h, "translation": t, "rotation": r, "subject": s, "trial_num": n}
        # TODO: want to check that you give a valid trial

        trial_folder = self.aruco_pics_loc / f"{h}_{t}_{r}_{s}_{n}"
        aruco_loc = self.af.full_analysis_single_id(trial_folder, aruco_id)

        aruco_loc.name = {"hand": h, "translation": t, "rotation": r, "subject": s, "trial_num": n}

        if save_trial:
            result_folder = self.aruco_data_loc / trial_name
            # print(f"Saved trial in: {result_folder}")
            aruco_loc.save_poses(file_name_overwrite=result_folder)

        return aruco_loc

    def load_calibration(self, calibration_loc, dist_loc):
        """
        Function loads camera calibration and dist values from csv files.
        """
        pass

    def batch_aruco_analysis(self, hand, exclude_tr_trials=True, include_rotation_only_trials=False, save_data=True,
                             assess_indices=False, crop_trial=False):
        """
        Runs aruco analysis on a set of trials. Collects data into a dictionary (trial_name) -> (trial data, as aruco_loc obj)
        """
        files_covered = list()
        batch_trial_data = dict()

        for s, h, t, r, n in datamanager.generate_all_names(subject=None, hand_name=hand,
                                                            exclude_tr_trials=exclude_tr_trials,
                                                            include_rotation_only_trials=include_rotation_only_trials):
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


if __name__ == '__main__':
    # remember, we have my_ast_files

    mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                    (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                    (0, 0, 1)))
    dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))

    marker_side_dims = 0.03  # in meters

    ar = AstArucoAnalysis(file_loc_obj=my_ast_files, camera_calib=mtx, camera_dists=dists, marker_side_dims=marker_side_dims)

    print("""
        You have run the script for:
            Aruco Analysis
             
        Please enter the number of the function you want:
        1) run analysis on one trial
        2) run analysis on a batch of trials
        3) view a trial like a video
        or enter "c" to cancel.
    """
    )
    option = input("What would you like to do? ")

    if option == "1":
        trial_option = input("Which trial would you like me to analyze? ( [hand]_[t]_[r]_[subject]_[trial_num] ), enter here: ")
        h, _, _, _, _ = trial_option.split("_")
        _, _, hand_id = get_hand_stats(h)
        ar.aruco_analyze_trial(trial_name=trial_option, aruco_id=2, save_trial=True)

    elif option == "2":
        hand_option = input("Which hand would you like me to analyze?")
        rot_option = input("What rotation condition would you like me to analyze?")
        _, _, hand_id = get_hand_stats(hand_option)
        ar.batch_aruco_analysis(hand=hand_option, save_data=True)

    elif option == "3":
        trial_option = input("Which trial would you like me to analyze?")
        h, t, r, s, n = trial_option.split("_")
        ad = datamanager.AstData(my_ast_files)

        ad.view_images(h, t, r, s, n)

    elif option == "c":
        # do nothing!
        pass

    else:
        print("invalid input!")











