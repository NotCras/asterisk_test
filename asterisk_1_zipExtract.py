from zipfile import ZipFile
from pathlib import Path
import asterisk_0_prompts as prompts
import asterisk_0_dataHelper as helper

def extract_file(folder, filename):
    file_name = folder + "/" + filename + ".zip"

    print("Extracting: " + filename)

    extract_folder = "viz/" + filename

    with ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        print("Completed Extraction.")

def batch_extract():
    subject_name = helper.collect_prompt_data(
        prompts.subject_name_prompt, prompts.subject_name_options)

    hand = helper.collect_prompt_data(
        prompts.hand_prompt, prompts.hand_options)

    folder_path = "asterisk_test_data/" + subject_name + "/" + hand

    print("FOLDER PATH")
    print(folder_path)

    #construct all the zip file names

    if hand == "basic" or hand == "m2stiff" or hand == "vf":
        types = ["none"]
    else:
        types = prompts.type_options

    for t in types:
        if t == "none":
            directions = prompts.dir_options
        else:
            directions = prompts.dir_options_no_rot

        for d in directions:
            for num in prompts.trial_options:

                zip_file = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num

                extract_file(folder_path, zip_file)

def single_extract():
    subject_name = helper.collect_prompt_data(
        prompts.subject_name_prompt, prompts.subject_name_options)

    hand = helper.collect_prompt_data(
        prompts.hand_prompt, prompts.hand_options)

    if hand == "basic" or hand == "m2stiff" or hand == "vf":
        types = ["none"]
    else:
        types = prompts.type_options

    t = helper.collect_prompt_data(
        prompts.type_prompt, types)

    if t == "none":
        directions = prompts.dir_options
    else:
        directions = prompts.dir_options_no_rot

    d = helper.collect_prompt_data(prompts.dir_prompt, directions)

    num = helper.collect_prompt_data(prompts.trial_prompt, prompts.trial_options)

    folder_path = "asterisk_test_data/" + subject_name + "/" + hand
    zip_file = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num

    print("FILENAME")
    print(zip_file)

    extract_file(folder_path, zip_file)


if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    print("""
    
        ========= ZIP FILE EXTRACTOR ==========
    I EXTRACT YOUR ZIP FILES FOR THE ASTERISK STUDY
        AT NO COST, STRAIGHT TO YOUR DOOR!
    
    """)
    input("PRESS <ENTER> TO CONTINUE.  ")
    
    mode = input("Do you want to extract a batch of files? [y/n] ")

    if mode == "y":
        batch_extract()
    else:
        single_extract()





    
