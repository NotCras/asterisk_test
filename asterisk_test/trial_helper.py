
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
import data_manager as prompts
from aruco_analysis import AstArucoAnalysis
from aruco_tool import ArucoFunc, ArucoLoc
import numpy as np
from file_manager import my_ast_files
from ast_hand_info import get_hand_stats
from ast_trial_translation import AstTrialTranslation


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
            prompts.type_prompt, prompts.type_options)

    return subject_name, hand, trial_type

def redo_prev_settings():
    subject_name = collect_prompt_data(
        prompts.subject_name_prompt, prompts.subject_name_options)
    hand = collect_prompt_data(prompts.hand_prompt, prompts.hand_options)

    trial_type = "none"  # if we are starting a new hand, we will definitely start with none
    print("Set trial type to `none' because new hand.")

    return subject_name, hand, trial_type

def choose_test(trial_type):
    if trial_type == "none":
        dir_label = collect_prompt_data(
            prompts.dir_prompt, prompts.dir_options)
    else:
        dir_label = collect_prompt_data(
            prompts.dir_prompt, prompts.dir_options_no_rot)

    trial_num = collect_prompt_data(
        prompts.trial_prompt, prompts.trial_options)

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

def full_camera_process(home, toFolder, zip_name):
    run_camera = True
    while run_camera:
        os.chdir(home)
        Path(toFolder).mkdir(parents=True)  # , exist_ok=True)
        os.chdir(toFolder)

        run_the_camera()

        print(" ")
        print("reminder: " + zip_name)
        print(" ")
        response = collect_prompt_data(
            prompts.check_prompt, prompts.check_options)

        if response == "yes":
                break
        else:
            remove_data()

            if response == "cancel":
                quit()
                break


def approve_new_data(home, data_folder, trial_name, thresholds):

    mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                    (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                    (0, 0, 1)))
    dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))

    marker_side_dims = 0.03  # in meters

    h, _, _, _, _ = trial_name.split("_")
    _, _, hand_id = get_hand_stats()

    aruco = AstArucoAnalysis(file_loc_obj=my_ast_files, camera_calib=mtx, camera_dists=dists,
                             marker_side_dims=marker_side_dims)

    # aruco analyze data you just collected
    path_al = aruco.aruco_analyze_trial(trial_name=trial_name, aruco_id=2, save_trial=False)

    # calculate metrics
    path = AstTrialTranslation(my_ast_files)
    path.add_data_by_arucoloc(path_al, norm_data=True, condition_data=True, do_metrics=True)

    # get best trial metrics


    # compare the two


    # output the findings as a report


    # generate plots


    pass


def remove_data():
    print("DELETING DATA")
    full_folder_path = Path.cwd()
    print(full_folder_path)
    shutil.rmtree(full_folder_path)

#=========================================================================
#============================ SCRIPT START ===============================
#=========================================================================
if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    subject_name, hand, trial_type = check_prev_settings()

    dir_label, trial_num = choose_test(trial_type)

    folder_path = "data/" + "/" + hand + "/" + dir_label + "/" + trial_type + "/" + subject_name + "/" + trial_num + "/"
    zipfile = hand + "_" + dir_label + "_" + trial_type + "_" + subject_name + "_" + trial_num

    print("FOLDER PATH")
    print(folder_path)

    full_camera_process(home_directory, folder_path, zipfile)
    
    print("COMPRESSING DATA")
    os.chdir(home_directory)

    # todo: log that we did this trial, double check if we are repeating trials based on the text input
    # note: currently script will error out if you enter the info for an existing trial... keeping as is

    shutil.make_archive(zipfile, 'zip', folder_path)
    print("COMPLETED TRIAL")
    print(zipfile)

        

