
'''
Code to...
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

#------------------------------------
subject_name_prompt = """
ENTER SUBJECTS NAME
(lowercase!)

Possible options:
"""
subject_name_options = ["john", "josh", "sage", "garth"]
subject_name = None

#------------------------------------
hand_prompt = """
ENTER HAND YOU ARE USING FOR THIS TRIAL
(lowercase!)

Possible options:
"""
hand_options = ["human", "barrett", "m2-passive", "m2-active", "modelo", "modelk", "basic", "modelvf", "2v2", "2v3", "3v3"]
hand = None

#------------------------------------
dir_prompt = """
ENTER DIRECTION OF CURRENT TRIAL
(lowercase!)

Possible options:
"""
dir_options = ["a", "b", "c", "d", "e", "f", "g", "h", "cw", "ccw"]
dir_options_no_rot = ["a", "b", "c", "d", "e", "f", "g", "h"]
dir_label = None

#------------------------------------
trial_prompt = """
WHAT NUMBER TRIAL IS THIS
(lowercase! ... :P)

Up to ...
"""
trial_options = ["1", "2", "3", "4", "5"] #, "6", "7", "8", "9", "10"]
trial_num = None

#------------------------------------
type_prompt = """
WHAT TYPE OF TRIAL IS THIS
(lowercase!)

Options ...
"""
type_options = ["none", "plus15", "minus15"]
trial_type = None

#------------------------------------
check_prompt = "Are you happy with this data? : "
check_options = ["yes", "no", "cancel"]

#------------------------------------
def check_prev_settings():
    try:
        answer, lines = check_temp_file()
    except:
        print("Did not find previous settings. Continuing...")
        answer = "n"

    if answer == "y":
        subject_name = lines[0]
        hand = lines[1]

    elif answer == "n":
        subject_name = collect_prompt_data(
            subject_name_prompt, subject_name_options)
        hand = collect_prompt_data(hand_prompt, hand_options)

        update_temp_file(subject_name, hand)

    else:
        quit()
    
    return subject_name, hand

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

def update_temp_file(subject, hand):
    with open(".asterisk_temp", 'w') as filetowrite:
        filetowrite.write(subject + '\n')
        filetowrite.write(hand)

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

#=========================================================================
#============================ SCRIPT START ===============================
#=========================================================================
home_directory = Path(__file__).parent.absolute()

subject_name, hand = check_prev_settings()

trial_type = collect_prompt_data(type_prompt, type_options)

if trial_type == "none":
    dir_label = collect_prompt_data(dir_prompt, dir_options)
else:
    dir_label = collect_prompt_data(dir_prompt, dir_options_no_rot)

trial_num = collect_prompt_data(trial_prompt, trial_options)


folder_path = "data/" + subject_name + "/" + hand + "/" + dir_label + "/" + trial_type + "/" + trial_num + "/"
zipfile = subject_name + "_" + hand + "_" + dir_label + "_" + trial_type + "_" + trial_num


print("FOLDER PATH")
print(folder_path)


run_camera = True
while run_camera:
    os.chdir(home_directory)
    Path(folder_path).mkdir(parents=True)#, exist_ok=True)
    os.chdir(folder_path)

    run_the_camera()

    print("reminder: " + zipfile)
    response = collect_prompt_data(check_prompt, check_options)

    if response == "yes":
        break
    else: 
        print("DELETING DATA")
        full_folder_path = Path.cwd()
        print(full_folder_path)
        shutil.rmtree(full_folder_path)

        if response == "cancel":
            quit()
            break

print("COMPRESSING DATA")
os.chdir(home_directory)

# todo: log that we did this trial, double check if we are repeating trials based on the text input
# note: currently script will error out if you enter the info for an existing trial... keeping as is

shutil.make_archive(zipfile, 'zip', folder_path)
#shutil.make_archive(zipfile, 'zip', subject_name)
print("COMPLETED TRIAL")
print(zipfile)

#collect files

    

