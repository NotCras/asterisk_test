
'''
Code to help run the asterisk test. Runs as a script, haven't made into oo class structure yet.
a) take in user input to determine data naming and file structure
        using var = input( "enter: ")
b) check that data was entered correctly - look for duplicates
c) make/go to correct folder
        using os.chdir(path)
d) run view_image image_saver ros node from inside the script
        using os.system("rosrun  ...")
e) compress the data into a zip file
        using shutil.make_archive(output_filename, 'zip', dir_name)

(make a log file that the script will check for whether a trial was run?)

'''

import os, shutil, keyboard, subprocess 
from curtsies import Input 
from pathlib import Path 
from aruco_analysis import AstArucoAnalysis
from aruco_tool import ArucoFunc, ArucoLoc
import numpy as np
from file_manager import my_ast_files
from ast_hand_info import get_hand_stats
from ast_trial_translation import AstTrialTranslation
from data_plotting import AsteriskPlotting


#------------------------------------

subject_name = None
hand = None
dir_label = None
trial_num = None
trial_type = None

#------------------------------------


def check_prev_settings():
    try:
        answer, lines = check_temp_file()
    except:
        print("Did not find previous settings. Continuing...")
        answer = "n"

    if answer == "y":
        subject_name, hand, trial_type = double_check_trial_type(lines)

    elif answer == "n":
        subject_name, hand, trial_type = redo_prev_settings()

    else:
        quit()
    
    update_temp_file(subject_name, hand, trial_type)
    
    return subject_name, hand, trial_type


def double_check_trial_type(lines):
    subject_name = lines[0]
    hand = lines[1]

    print("Previous asterisk type: " + lines[2])
    type_ans = input("Is this still correct? [y/n/c]  ")

    if(type_ans == "y"):
        trial_type = lines[2]
    else:
        print("Then please enter the new asterisk type.")
        trial_type = collect_prompt_data(
            type_prompt, type_options)

    return subject_name, hand, trial_type


def redo_prev_settings():
    subject_name = collect_prompt_data(
        subject_name_prompt, subject_name_options)
    hand = collect_prompt_data(hand_prompt, hand_options)

    trial_type = "none"  # if we are starting a new hand, we will definitely start with none
    print("Set trial type to `none' because new hand.")

    return subject_name, hand, trial_type


def choose_test(trial_type):
    if trial_type == "x":
        dir_label = collect_prompt_data(
            dir_prompt, dir_options)
    else:
        dir_label = collect_prompt_data(
            dir_prompt, dir_options_no_rot)

    trial_num = collect_prompt_data(
        trial_prompt, trial_options)

    return dir_label, trial_num


def check_temp_file():
    with open('.asterisk_temp') as f:
        print("Found previous settings.")
        lines = [line.rstrip() for line in f]

    if lines:
        print("Previous settings:   " + lines[0] + ", " + lines[1])
        answer = input("Is this still correct? [y/n/c]  ")

    else:
        answer = "n"

    return answer, lines


def update_temp_file(subject, hand, trial):
    with open(".asterisk_temp", 'w') as filetowrite:
        filetowrite.write(subject + '\n')
        filetowrite.write(hand + '\n')
        filetowrite.write(trial)

    print("Updated settings.")


def collect_prompt_data(prompt, options):
    print(prompt)
    for opt in options:
        print(opt)
    print("   ")

    for x in range(3): #number of tries
        variable = input("ENTER HERE: ")

        if variable in options:
            break

        else:
            print("INVALID INPUT. TRY AGAIN.")
            variable = "INVALID"

    if variable == "INVALID":
        print("TIMED OUT. TRY AGAIN FROM SCRATCH.")
        quit()

    print("======================================== ")
    return variable


def run_the_camera():
    print("   ")
    print("Ready to run camera...")
    print("   ")
    print("PRESS <SPACE> TO STOP CAMERA WHEN RUNNING")
    print("   ")
    input("Press <ENTER>, when ready, to start the camera")
    print("CAMERA STARTED")

    camera_cmd = "rosrun image_view image_saver image:=/camera/color/image_raw"

    a = subprocess.Popen("exec " + camera_cmd, shell=True)

    waiting = True
    with Input() as input_generator:
        for c in input_generator:
            while waiting:
                print(c)

                if c == '<SPACE>':
                    a.terminate()
                    print("KILLING CAMERA PROCESS")
                    waiting = False

            break


def full_camera_process(file_obj, trial_name, metrics_and_thresholds):
    run_camera = True
    while run_camera:
        #os.chdir(home)
        #Path(toFolder).mkdir(parents=True)  # , exist_ok=True)
        pics_path = file_obj.aruco_pics / trial_name
        os.chdir(pics_path)

        run_the_camera()

        approve_new_data(file_obj, trial_name, metrics_and_thresholds)

        print(" ")
        print("reminder: " + trial_name)
        print(" ")
        response = collect_prompt_data(
            check_prompt, check_options)

        if response == "yes":
                break
        else:
            remove_data(file_obj, trial_name)

            if response == "cancel":
                quit()
                break


def approve_new_data(file_obj, trial_name, metrics_and_thresholds):

    mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                    (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                    (0, 0, 1)))
    dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))
    marker_side_dims = 0.03  # in meters

    h, _, _, _, _ = trial_name.split("_")
    _, _, hand_id = get_hand_stats()

    aruco = AstArucoAnalysis(file_loc_obj=file_obj, camera_calib=mtx, camera_dists=dists,
                             marker_side_dims=marker_side_dims)

    # aruco analyze data you just collected
    path_al = aruco.aruco_analyze_trial(trial_name=trial_name, aruco_id=2, save_trial=False)

    # calculate metrics
    path = AstTrialTranslation(file_obj)
    path.add_data_by_arucoloc(path_al, norm_data=True, condition_data=True, do_metrics=True)

    # get best trial metrics
    dict_of_best_trials = file_obj.data_home / "best_trials.csv"
    best_trial = get_best_trial(file_obj,
                                path.hand.hand_name, path.trial_translation, path.trial_rotation,
                                dict_of_best_trials)

    # compare the two sets of metrics
    # which metrics do we want to compare?
    # total_distance, mvt_efficiency, max_error, max_rotation error
    # also can calculate the frechet distance between the two lines
    print(f"=========================================")
    print(f" Metric Comparison results for {trial_name}")
    metrics, final = compare_paths(path, best_trial, metrics_and_thresholds)

    # generate plots
    AsteriskPlotting.compare_paths(best_trial, path)


def get_best_trial(file_obj, hand, direction, rotation, best_trial_dict=None):
    """
    Retrieves the file_name of the best trial. Data needs to have been saved already in aruco_data location.
    If no best trial exists, then returns ideal line.
    """
    best_trial = AstTrialTranslation(file_obj)

    best_trial_key = f"{hand}_{direction}"  # TODO: add rotation, so its hand_direction_rotation
    try:
        best_trial_name = best_trial_dict[best_trial_key]
        best_trial.add_data_by_file(best_trial_name)

    except KeyError or TypeError:
        # if key doesn't exist, or we didn't get a dict, then we get the ideal line to compare to
        pass  # TODO: if no best trial exists, replace with ideal line... *actually, do we want this like this? Revisit

    return best_trial


def metric_comparison(best_value, path_value, threshold, direction="band"):
    """
    Compares the metrics of the path that you just took to the best trial. Returns True or False given if path value
    falls within acceptable range of best_value.

    threshold - acceptable percent error
    direction (-1, 0, 1) - indicates whether low-pass ("low"), high-pass ("high"), or band-pass ("band")

    Band-pass will respect the threshold up and down (+/- threshold).
    Low-pass will only respect upper threshold (+ threshold and lower is allowed)
    High-pass will only respect lower threshold (- threshold and higher is allowed)
    """

    # need to have thresholds for each metric, save in a dict
    percent_error = best_value - path_value
    percent_error /= best_value

    if direction is "high":
        fit = percent_error >= -threshold

    elif direction is "low":
        fit = percent_error <= threshold

    else:  # direction is band
        fit = threshold >= percent_error >= -threshold

    return fit


def compare_paths(best_trial_obj, path_trial_obj, metrics={}):
    """
    Compares the path trial to the best trial at the metrics specified
    Metrics is a dict where the key is the metric name, and the value is a tuple (threshold, direction)
    Will always compute frechet distance between best trial and path trial
    """
    best_trial_metric_values = best_trial_obj.metrics
    path_trial_metric_values = path_trial_obj.metrics

    metric_recommendations = {}

    for m in metrics.keys():
        finding = metric_comparison(best_trial_metric_values[m], path_trial_metric_values[m],
                                    metrics[m][0], metrics[m][1])

        print(f"{m}: {best_trial_metric_values[m]} v {path_trial_metric_values[m]}      | {finding}")
        metric_recommendations[m] = finding

    # frechet distance calculation
    fd = 0  # TODO: make calculations happen
    fd_finding = 0.2 >= fd >= -0.2

    print(f"fd: {fd}    |{fd_finding}")
    metric_recommendations["fd"] = fd

    # make final recommendation, if all are within the thresholds
    final = all(i for i in list(metric_recommendations))
    print(f"Final Recommendation: {final}")
    print(f"   ")

    return metric_recommendations, final


def remove_data(file_obj, trial_name):
    print("DELETING DATA")
    pics_path = file_obj.aruco_pics / trial_name
    print(pics_path)
    shutil.rmtree(pics_path)


#
# because I'm pressed for time, I'm adding this below. Will code this out later by properly using data manager
#------------------------------------
subject_name_prompt = """
ENTER SUBJECTS NAME
(lowercase!)

Possible options:
"""
subject_name_options = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]

#------------------------------------
hand_prompt = """
ENTER HAND YOU ARE USING FOR THIS TRIAL
(lowercase!)

Possible options:
"""
hand_options = ["1v1", "2v1", "p1vp1", "2v2", "2v3", "3v3", "p2vp2"]

#------------------------------------
dir_prompt = """
ENTER DIRECTION OF CURRENT TRIAL
(lowercase!)

Possible options:
"""
dir_options = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "+", "-"]
dir_options_no_rot = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

#------------------------------------
trial_prompt = """
WHAT NUMBER TRIAL IS THIS
(lowercase! ... :P)

Up to ...
"""
trial_options = ["1", "2", "3"] #, "4", "5"]  # , "6", "7", "8", "9", "10"]

#------------------------------------
type_prompt = """
WHAT TYPE OF TRIAL IS THIS
(lowercase!)

Options ...
"""
type_options = ["x", "+15", "-15"]

#------------------------------------
check_prompt = "Are you happy with this data? : "
check_options = ["yes", "no", "cancel"]

temp_file_check = "Are you still doing"





# =========================================================================
# ============================ SCRIPT START ===============================
# =========================================================================
if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    subject_name, hand, trial_type = check_prev_settings()

    dir_label, trial_num = choose_test(trial_type)

    trial_name = f"{hand}_{dir_label}_{trial_type}_{subject_name}_{trial_num}"

    metrics_to_check = {"dist": (0.2, "high"),
                        "mvt_eff": (0.2, "high"),
                        "max_err": (0.2, "low"),
                        "max_err_rot": (0.2, "low")
                        }

    # folder_path = "data/" + "/" + hand + "/" + dir_label + "/" + trial_type + "/" + subject_name + "/" + trial_num + "/"
    # zipfile = hand + "_" + dir_label + "_" + trial_type + "_" + subject_name + "_" + trial_num

    print("FOLDER PATH")
    print(my_ast_files.aruco_pics / trial_name)

    full_camera_process(my_ast_files, trial_name, metrics_to_check)
    
    print("COMPRESSING DATA")
    os.chdir(home_directory)

    # todo: log that we did this trial, double check if we are repeating trials based on the text input
    # note: currently script will error out if you enter the info for an existing trial... keeping as is

    compress_path = my_ast_files.compressed_data
    shutil.make_archive(trial_name, 'zip', compress_path)
    print(f"COMPLETED TRIAL: {trial_name}")
    print("  ")

        

