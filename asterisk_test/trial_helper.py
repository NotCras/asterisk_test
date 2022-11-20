
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

import os, shutil, keyboard, subprocess, csv
import pdb
import similaritymeasures as sm
import pandas as pd
from curtsies import Input 
from pathlib import Path 
import numpy as np

from aruco_analysis import AstArucoAnalysis
from aruco_tool import ArucoFunc, ArucoLoc
from file_manager import AstDirectory, my_ast_files
from ast_hand_info import get_hand_stats
from ast_trial_translation import AstTrialTranslation
from ast_trial_rotation import AstTrialRotation
from data_plotting import AsteriskPlotting
from data_manager import AstData


# for best trial data
data_directory = Path("/home/shakey/Desktop/best_trials")
best_trial_loc = AstDirectory()
best_trial_loc.data_home = data_directory
best_trial_loc.aruco_pics = data_directory / "viz"
best_trial_loc.aruco_data = data_directory / "aruco_data"
best_trial_loc.metric_results = data_directory / "results"
best_trial_loc.result_figs = data_directory / "results" / "plots"
best_trial_loc.debug_figs = data_directory / "results" / "debug_plots"
best_trial_loc.resources = data_directory.parent / "resources"


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

    if hand in ["2v1", "p1vp1"]:
        trial_type = "x"
    else:		
        trial_type = collect_prompt_data(type_prompt, type_options)

    return subject_name, hand, trial_type


def choose_test(trial_type):
    if trial_type == "x":
        dir_label = collect_prompt_data(
            dir_prompt, dir_options)
    else:
        dir_label = collect_prompt_data(
            dir_prompt, dir_options_no_rot)
    
    if dir_label == "x":
        rot_label = collect_prompt_data(
            rot_prompt, dir_rot_only)
    else:
        rot_label = trial_type

    trial_num = collect_prompt_data(
        trial_prompt, trial_options)

    return dir_label, rot_label, trial_num
    
    
def choose_rot():

    rot_label = collect_prompt_data(
        rot_prompt, dir_rot_only)

    return rot_label


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

    print(" |   ok! ")
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


def full_camera_process(file_obj, trial_name, metrics_and_thresholds, best_loc=None):
    h, t, r, s, n = trial_name.split("_")

    # get best trial metrics
    print("============================")
    print("FIRST, I'M GETTING THE BEST TRIAL READY")
    
    if best_loc is not None:
        best_trials_list_loc = best_loc.data_home / "top_trials1_all.csv"
        dict_of_best_trials = read_best_trials(best_trials_list_loc)
        best_trial = get_best_trial(best_loc,
                                    h, t, r, True,
                                    dict_of_best_trials)
    else:
        best_trials_list_loc = file_obj.data_home / "top_trials1_all.csv"
        dict_of_best_trials = read_best_trials(best_trials_list_loc)
        best_trial = get_best_trial(file_obj,
                                    h, t, r, True,
                                    dict_of_best_trials)

    best_name = best_trial.generate_name()
    _, _, _, b_s, b_n = best_name.split("_")
    old_h, old_t, old_r = convert_notation_new_to_old(h, t, r)

    print("Ready to view the best trial?")
    print("Hit space+enter to watch. Hit s+enter to skip. Hit q+enter to stop.")
    char = input(" ")
    print("")

    if char == "q":
        quit()
    elif char == "s":
        print("Skipping best trial!")
    else:    
    # view the best trial, repeat as needed
        view_best_trial = True
        while view_best_trial:
            if best_loc is not None:
                manager = AstData(best_loc)
            else:
                manager = AstData(my_ast_files)

            manager.view_images_light(old_h, old_t, old_r, b_s, b_n, do_quit=False)
        
            response = collect_prompt_data(
                view_check_prompt, check_options)

            if response == "no":
                view_best_trial = False
            else:
                if response == "cancel":
                    quit()
    
    print("============================")
    print("NOW IT'S TIME TO COLLECT DATA!")
    print(" ")
    
    collect_data = True
    while collect_data:
        print(" ")
        print(f"REMEMBER: {trial_name}")
        print(" ")
        # os.chdir(home)
        # Path(toFolder).mkdir(parents=True)  # , exist_ok=True)
        pics_path = file_obj.aruco_pics / trial_name
        os.mkdir(pics_path)
        os.chdir(pics_path)

        run_the_camera()

        print("============================")
        print("NOW WE WILL PROCESS THE DATA WE JUST TOOK!")
        print(" ")
		
        if t == "x":
            approve_new_data_rot(file_obj, trial_name, best_trial)
        else:
            approve_new_data(file_obj, trial_name, best_trial, metrics_and_thresholds)

        response = collect_prompt_data(
            check_prompt, check_options)

        if response == "yes":
            collect_data = False

        else:
            remove_data(file_obj, trial_name)

            if response == "cancel":
                quit()


def approve_new_data(file_obj, trial_name, best_trial, metrics_and_thresholds):
    hand_to_id = {"1v1": 0, "2v1": 2, "p1vp1": 1, "2v2": 3, "2v3": 4, "3v3": 5, "p2vp2": 6}

    mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                    (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                    (0, 0, 1)))
    dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))
    marker_side_dims = 0.03  # in meters

    h, _, _, _, _ = trial_name.split("_")
    _, _, hand_id = get_hand_stats(h)

    aruco = AstArucoAnalysis(file_loc_obj=file_obj, camera_calib=mtx, camera_dists=dists,
                             marker_side_dims=marker_side_dims)

    # aruco analyze data you just collected
    path_al = aruco.aruco_analyze_trial(trial_name=trial_name, aruco_id=hand_to_id[h], save_trial=False)

    # calculate metrics
    path = AstTrialTranslation(file_obj)
    path.add_data_by_arucoloc(path_al, norm_data=True, condition_data=True, do_metrics=True)

    # compare the two sets of metrics
    # which metrics do we want to compare?
    # total_distance, mvt_efficiency, max_error, max_rotation error
    # also can calculate the frechet distance between the two lines
    print(f"=========================================")
    print(f"Metric Comparison results for {trial_name}")
    metrics, final = compare_paths(path, best_trial, metrics_and_thresholds)

    # generate plots
    AsteriskPlotting.compare_paths(best_trial, path)


def approve_new_data_rot(file_obj, trial_name, best_trial):
    hand_to_id = {"1v1": 0, "2v1": 2, "p1vp1": 1, "2v2": 3, "2v3": 4, "3v3": 5, "p2vp2": 6}

    mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                    (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                    (0, 0, 1)))
    dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))
    marker_side_dims = 0.03  # in meters

    h, _, r, _, _ = trial_name.split("_")
    _, _, hand_id = get_hand_stats()

    aruco = AstArucoAnalysis(file_loc_obj=file_obj, camera_calib=mtx, camera_dists=dists,
                             marker_side_dims=marker_side_dims)

    # aruco analyze data you just collected
    path_al = aruco.aruco_analyze_trial(trial_name=trial_name, aruco_id=hand_to_id[h], save_trial=False)

    # calculate metrics
    path = AstTrialRotation(file_obj)
    path.add_data_by_arucoloc(path_al, norm_data=True, condition_data=True, do_metrics=True)

    # compare the two sets of metrics
    # which metrics do we want to compare?
    # total_distance, mvt_efficiency, max_error, max_rotation error
    # also can calculate the frechet distance between the two lines
    print(f"=========================================")
    print(f"Metric Comparison results for {trial_name}")

    # get max rotation
    _, _, b_path_ang = best_trial.get_poses(use_filtered=False)
    _, _, p_path_ang = path.get_poses(use_filtered=False)
    b_max = max(np.abs(b_path_ang))
    p_max = max(np.abs(p_path_ang))
    fit, perc_err = metric_comparison(p_max, b_max, 0.2, direction="high")
    print(f"{r}: {b_max} v {p_max}: {perc_err}     | {fit}")


def read_best_trials(best_trial_file):
    # from: https://stackoverflow.com/questions/33858989/how-to-read-a-csv-into-a-dictionary-in-python

    df = pd.read_csv(best_trial_file)
    df = df.set_index('key')

    return df.to_dict()['best_trial']


def convert_notation_new_to_old(hand, direction, rotation):
    t_dict = {"no": "a", "ne": "b", "ea": "c", "se": "d", "so": "e", "sw": "f", "we": "g", "nw": "h", "x": "n"}
    r_dict = {"pp": "cw", "mm": "ccw", "m15": "m15", "p15": "p15", "x": "n"}
    h_dict = {"1v1": "basic", "2v1": "m2active", "p1vp1": "palm1r",
              "2v2": "2v2", "2v3": "2v3", "3v3": "3v3", "p2vp2": "palm2r"}
    #pdb.set_trace()
    #if direction in ["pp", "mm"]:
    #    c_d = "n"
    #    c_r = r_dict[direction]
    #else:
    c_d = t_dict[direction]
    c_r = r_dict[rotation] # TODO: funky business happening here now when rotation only trial
		
    return h_dict[hand], c_d, c_r


def get_best_trial(file_obj, hand, direction, rotation, convert_notation=True, best_trial_dict=None):
    """
    Retrieves the file_name of the best trial. Data needs to have been saved already in aruco_data location.
    """
    if direction in ["no", "ne", "ea", "se", "so", "sw", "we", "nw"]:
        best_trial = AstTrialTranslation(file_obj)
        
    else:
        best_trial = AstTrialRotation(file_obj)

    if convert_notation:
        hand, direction, rotation = convert_notation_new_to_old(hand, direction, rotation)

    best_trial_key = f"{hand}_{direction}_{rotation}"

    best_trial_name = best_trial_dict[best_trial_key]
    best_trial.load_data_by_aruco_file(best_trial_name + ".csv", old=False)

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

    if direction == "high":
        fit = percent_error >= -threshold

    elif direction == "low":
        fit = percent_error <= threshold

    else:  # direction is band
        fit = threshold >= percent_error >= -threshold

    return fit, percent_error


def fd_best(best_trial_obj, path_trial_obj):
    """
    Calculate the frechet distance between self.poses and a target path
    Uses frechet distance calculation from asterisk_calculations object
    """
    b_x, b_y, b_path_ang = best_trial_obj.get_poses(use_filtered=False)
    b_path_t = np.column_stack((b_x, b_y))
    p_x, p_y, p_path_ang = path_trial_obj.get_poses(use_filtered=False)
    p_path_t = np.column_stack((p_x, p_y))

    try:
        t_fd = sm.frechet_dist(b_path_t, p_path_t)  # only get translation from target line
        # pdb.set_trace()
        # r_fd = sm.frechet_dist(o_path_ang, ast_trial.target_rotation)  # just max error right now
        # fd = AstMetrics.calc_frechet_distance_all(ast_trial)

    except Exception as e:
        print(f"{path_trial_obj.generate_name()}, Frechet Distance failure.")
        print(e)
        t_fd = -1

    return t_fd


def compare_paths(best_trial_obj, path_trial_obj, metrics={}):
    """
    Compares the path trial to the best trial at the metrics specified
    Metrics is a dict where the key is the metric name, and the value is a tuple (threshold, direction)
    Will always compute frechet distance between best trial and path trial
    """
    best_trial_metric_values = best_trial_obj.metrics
    path_trial_metric_values = path_trial_obj.metrics

    metric_recommendations = dict()
    
    if best_trial_metric_values is None:
        best_trial_obj.update_all_metrics()
        best_trial_metric_values = best_trial_obj.metrics

    print("Suggestion | metric: percent_error <- best_trial vs current_trial")
    for m in metrics.keys():
        finding, perc_err = metric_comparison(best_trial_metric_values[m], path_trial_metric_values[m],
                                    metrics[m][0], metrics[m][1])

        print(f"{finding}    | {m}: {round(perc_err, 2)}           <- {round(best_trial_metric_values[m], 3)} v {round(path_trial_metric_values[m], 3)} | {metrics[m][0]}, {metrics[m][1]}  ")
        metric_recommendations[m] = finding

    # frechet distance calculation
    fd = fd_best(best_trial_obj, path_trial_obj)
    fd_finding = 0.4 >= fd >= -0.4

    print(f"{fd_finding}    | fd: {fd}   ")
    metric_recommendations["fd"] = fd
    metric_recommendations["debug"] = True # this extra one avoids issue where all recs are False

    # make final recommendation, if all are within the thresholds
    final = all(i for i in list(metric_recommendations))
    print("-----------")
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
subject_name_options = ["s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22"]

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
dir_options = ["no", "ne", "ea", "se", "so", "sw", "we", "nw", "x"]
dir_options_no_rot = ["no", "ne", "ea", "se", "so", "sw", "we", "nw"]

#------------------------------------
rot_prompt = """
ENTER ROTATION DIRECTION OF CURRENT TRIAL
(lowercase!)

Possible options:
"""
dir_rot_only = ["pp", "mm"]

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
type_options = ["x", "p15", "m15"]

#------------------------------------
check_prompt = "Are you happy with this data? : "
check_options = ["yes", "no", "cancel"]
view_check_prompt = "Do you want to watch again? : "

temp_file_check = "Are you still doing"


def trial_full_round(hand, dir_label, trial_type, subject_name, trial_num):
    trial_name = f"{hand}_{dir_label}_{trial_type}_{subject_name}_{trial_num}"
    
    print("|=============================|")
    print("|-----------------------------|")
    print("|------ DATA COLLECTION ------|")
    print("|-----------------------------|")
    print("")
    print(f"        {trial_name}       ")
    print("|=============================|")
    print(" ")
    print(f"      {dir_label}, {trial_type}       ")
    print("|=============================|")
    print(" ")

    metrics_to_check = {"dist": (0.25, "high"),
                        "mvt_eff": (0.5, "high"),
                        "max_err": (1.0, "band"),
                        "max_err_rot": (5.0, "band")
                        }

    print("LOOKING AT THIS FOLDER PATH")
    print(my_ast_files.aruco_pics / trial_name) 

    full_camera_process(my_ast_files, trial_name, metrics_to_check, best_loc=best_trial_loc)

    print("")
    print("============================")
    print("COMPRESSING DATA!")
    os.chdir(home_directory)

    # todo: log that we did this trial, double check if we are repeating trials based on the text input
    # note: currently script will error out if you enter the info for an existing trial... keeping as is

    folder_path = my_ast_files.aruco_pics / trial_name
    new_data_path = Path("/media/shakey/16ABA159083CA32B/new_data")
    #compress_path = my_ast_files.compressed_data
    full_compress_path = new_data_path / "compressed" / trial_name
    full_viz_path = new_data_path / "viz" / trial_name
    
    shutil.make_archive(str(full_compress_path), 'zip', str(folder_path))
    print(f"Compressed: {full_compress_path}")
    
    shutil.move(folder_path, full_viz_path)
    print(f"Moved uncompressed data to hard drive: {full_viz_path}")

    print("============================")
    print("  ")



# =========================================================================
# ============================ SCRIPT START ===============================
# =========================================================================
if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    subject_name, hand, temp_trial_type = check_prev_settings() 

    # ["no", "ne", "ea", "se", "so", "sw", "we", ]
    # "no", "ne", "nw", "ea", "we", "so", "se", "sw",
    dir_options = ["no", "ne", "nw", "ea", "we", "so", "se", "sw", "x"]
    dir_options_no_rot = ["no", "ne", "nw", "ea", "we", "so", "se", "sw"]
    
    if hand in ["2v1", "p1vp1"]:
        directions = dir_options_no_rot
        # check_prev_settings already handles the cases where type might be m15 or p15
        
    else:		
        if temp_trial_type in ["m15", "p15"]:
            directions = dir_options_no_rot
        else:
            directions = dir_options

    for d in directions:
        if d == "x":
            trial_type = ["pp", "mm"]
        else:
            trial_type = [temp_trial_type]		
        
        for r in trial_type:
            trial_options = ["1","2","3"]
            for n in trial_options:
                trial_full_round(hand, d, r, subject_name, n)
                #print(f"{hand}_{d}_{r}_{subject_name}_{n}")
            
            print(f"Completed direction {d}!")
            print(" ")
            print("TIME TO DO THE SURVEY!")
            print("  ")
            print("Get ready to move on! Hit space+enter to continue. Hit q+enter to stop.")
            char = input(" ")
            print("")
            os.system('cls' if os.name=='nt' else 'clear')
    
            if char == "q":
                quit()

    

