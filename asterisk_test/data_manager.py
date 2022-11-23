#!/usr/bin/env python3

"""
Handles various data management classes:
1) AstData: handles extracting compressed trial data
2) AstNaming: holds all of the relevant trial options
3) generate options function - a function that uses the AstNaming class to return any sort of list of options
4) generate_... - a set of four one-stop-shop generator functions for iterating over all of the trial options
5) smart input function: function which is key to the prompts one will find if they run most of the files as scripts
6) AstDir: handles directory locations for the rest of the classes (NOT DONE YET)
"""

import os
import pdb

import matplotlib.pyplot as plt

from pathlib import Path
from zipfile import ZipFile

from file_manager import my_ast_files


class AstData:
    def __init__(self, file_loc_obj):
        """
        Class which contains helper functions for data wrangling - getting ready for asterisk data analysis
        :param home: home directory of git repo
        """
        self.home = Path(__file__).parent.absolute()
        self.file_loc = file_loc_obj

    def view_images(self, hand_name, translation_name, rotation_name, subject_name, trial_number, do_quit=True):
        """
        View images of trial specified as a video
        :param subject_name: name of subject
        :param hand_name: name of hand
        :param translation_name: name of direction
        :param rotation_name: name of rotation
        :param trial_number: trial number
        """
        os.chdir(self.home)
        data_name = f"{subject_name}_{hand_name}_{translation_name}_{rotation_name}_{trial_number}"
        file_dir = self.file_loc.aruco_pics / data_name
        os.chdir(file_dir)

        files = [f for f in os.listdir('.') if f[-3:] == 'jpg']
        files.sort()

        img = None
        for f in files:
            im = plt.imread(f)

            if img is None:
                img = plt.imshow(im)
            else:
                img.set_data(im)

            plt.pause(.01)
            plt.draw()

        os.chdir(self.home)
        repeat = smart_input("Show again? [y/n]", "consent")
        if repeat == "y":
            # run again

            self.view_images(hand_name, translation_name,
                             rotation_name, subject_name, trial_number)
        else:
            # stop running
            if do_quit:
                quit()


    def view_images_light(self, hand_name, translation_name, rotation_name, subject_name, trial_number, do_quit=True):
        """
        View images of trial specified as a video
        :param subject_name: name of subject
        :param hand_name: name of hand
        :param translation_name: name of direction
        :param rotation_name: name of rotation
        :param trial_number: trial number
        """
        #os.chdir(self.home)
        data_name = f"{subject_name}_{hand_name}_{translation_name}_{rotation_name}_{trial_number}"
        file_dir = self.file_loc.aruco_pics / data_name
        #os.chdir(file_dir)

        #pdb.set_trace()
        files = [f for f in os.listdir(file_dir) if f[-3:] == 'jpg']
        files.sort()

        img = None
        for f in files:
            full_f = file_dir / f
            im = plt.imread(full_f)

            if img is None:
                img = plt.imshow(im)
            else:
                img.set_data(im)

            plt.pause(.01)
            plt.draw()


        plt.pause(1)
        plt.close()
        #os.chdir(self.home)
#        repeat = smart_input("Show again? [y/n]", "consent")
#        if repeat == "y":
#            # run again
#
#            self.view_images_light(hand_name, translation_name,
#                             rotation_name, subject_name, trial_number, do_quit=do_quit)
#        else:
#            # stop running
#            if do_quit:
#                quit()

    def single_extract(self, subject_name, hand_name, translation_name, rotation_name, trial_number):
        """
        Extract a single zip file, specified by parameters.
        :param subject_name: name of subject
        :param hand_name: name of hand
        :param translation_name: name of direction
        :param rotation_name: name of rotation
        :param trial_number: trial number
        """
        folders = f"compressed_data/{subject_name}/{hand_name}/"
        file_name = f"{subject_name}_{hand_name}_{translation_name}_{rotation_name}_{trial_number}"

        extract_from = folders+file_name+".zip"

        extract_to = f"viz/{file_name}"

        with ZipFile(extract_from, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Completed Extraction: {extract_to}")

    def batch_extract(self, subject_name, hand_name):
        """
        Extract a batch of zip files for a specific subject and hand
        :param subject_name: name of subject
        :param hand_name: name of hand
        """
        for s, h, t, r, n in generate_names_with_s_h(subject_name, hand_name):
            self.single_extract(s, h, t, r, n)



# TODO: move following functions into a AstNaming object?
def get_option_list(key):
    """
    One function to return all sorts of parameter lists. Mainly to be used outside of data manager
    :param key: the key of the list that you want
    :return: list of parameters
    """
    # "1v1": "basic", "2v1": "m2active", "p1vp1": "palm1r", "2v2": "2v2", "2v3": "2v3", "3v3": "3v3", "p2vp2": "palm2r"
    # "no": "a", "ne": "b", "ea": "c", "se": "d", "so": "e", "sw": "f", "we": "g", "nw": "h", "x": "n"

    options = {
        "hands": ["2v1","p1vp1", "2v2", "2v3", "3v3", "p2vp2"],  #["basic", "m2active", "palm1r", "2v2", "2v3", "3v3", "palm2r"],  #["m2stiff", "modelvf"]
        "subjects": ["s11", "s12", "s13", "s14", "s15", "s16"], #["sub1", "sub2", "sub3"],  #["s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22"]
        "translation_options": ["no", "ne", "ea", "se", "so", "sw", "we", "nw", "x"],  #["a", "b", "c", "d", "e", "f", "g", "h", "x"],
        "translation_only": ["no", "ne", "ea", "se", "so", "sw", "we", "nw"],  #["a", "b", "c", "d", "e", "f", "g", "h"],
        "rotation_only": ["pp", "mm"],  #["cw", "ccw"],  #
        "rotation_types": ["x", "p15", "m15"]  #["n", "p15", "m15"]  #
    }

    # opt = AstNaming()  # TODO: revisit this!!!
    return options[key]


def generate_t_r_pairs(hand_name, exclude_tr_trials=False, include_rotation_only_trials=True):
    """
    Generator that feeds all trial combinations pertaining to a specific hand
    :param hand_name: name of hand specified
    :return: yields translation and rotation combinations
    """
    translations = get_option_list("translations_all")
    n_trans_rot_opts = get_option_list("rotations_n_trans")
    rotations = get_option_list("rotations")

    for t in translations:
        if t == "n" and include_rotation_only_trials:  # necessary to divide rotations because cw and ccw only happen with no translation
            if hand_name in get_option_list("hands_only_n"):
                # if the hand can't do rotations, then don't yield this combination
                continue
            else:
                # otherwise include the rotation_only trials
                rot = n_trans_rot_opts

        elif t == "n" and not include_rotation_only_trials:
            # if we don't include rotation only, then we don't yield this combination
            continue

        else:
            # if we have a straightforward translation...
            if hand_name in get_option_list("hands_only_n") or exclude_tr_trials:
                # if the hand can't do rotations or we don't want to yield translation+rotation conditions, then we only consider translation-only trials
                rot = "n"
            else:
                rot = rotations

        # based on the t value and the what the hand can do and what the user specified, yield the salient t, r combinations
        for r in rot:
            yield t, r


def generate_names_with_s_h(subject_name, hand_name, exclude_tr_trials=False, include_rotation_only_trials=True):
    """
    Generates all trial combinations with a specific hand name and subject
    :param subject_name: name of subject
    :param hand_name: name of hand
    :return: yields all parameters
    """
    num = get_option_list("numbers")

    for t, r in generate_t_r_pairs(hand_name, exclude_tr_trials=exclude_tr_trials, include_rotation_only_trials=include_rotation_only_trials):
        for n in num:
            yield subject_name, hand_name, t, r, n

def generate_names_with_h(hand_name, exclude_tr_trials=False, include_rotation_only_trials=True):
    """
    Generates all trial combinations with a specific hand name
    :param subject_name: name of subject
    :param hand_name: name of hand
    :return: yields all parameters
    """
    for s in get_option_list("subjects"):
        for t, r in generate_t_r_pairs(hand_name, exclude_tr_trials=exclude_tr_trials, include_rotation_only_trials=include_rotation_only_trials):
            for n in get_option_list("numbers"):
                yield s, hand_name, t, r, n

def generate_all_names(subject=None, hand_name=None, exclude_tr_trials=False, include_rotation_only_trials=False):
    """
    Generate all combinations of all parameters
    :param subject: list of subjects to provide, if none provided defaults to all subjects
    :param hand_name: list of hands to provide, if none provided defaults to all hands
    :return: yields all combinations specified
    """
    # TODO: make smart version so you can be picky with your options... make the constant lists as default parameters
    if subject is None:
        subject = get_option_list("subjects")

    if hand_name is None:
        hand_name = get_option_list("hands")

    for s in subject:
        for h in hand_name:
            yield generate_names_with_s_h(s, h, exclude_tr_trials=exclude_tr_trials, include_rotation_only_trials=include_rotation_only_trials)


def generate_fname(subject_name, hand):
    """Create the full pathname
    # :param folder_path Directory where data is located -> currently not used
    :param subject_name Name of subject
    :param hand Name of hand"""

    for s, h, t, r, n in generate_names_with_s_h(subject_name, hand):
        file_name = f"{s}_{h}_{t}_{r}_{n}.csv"

        # total_path = folder_path + file_name
        # yield total_path
        yield file_name


def smart_input(prompt, option, valid_options=None):
    """
    Asks for input and continues asking until there is a valid response
    :param prompt: the prompt that you want printed
:param option: the option you want the input to choose from,
        if not in the options will look at valid_options for option
    :param valid_options: provides the ability to specify your own custom options
    """
    values = {  # TODO: make this use generate_options
        "subjects": ["sub1", "sub2", "sub3"],
        "hands": ["2v2", "2v3", "3v3", "barrett", "basic", "human", "m2active", "m2stiff", "modelvf"],
        "translations": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "translations_w_n": ["a", "b", "c", "d", "e", "f", "g", "h", "n"],
        "rotation_combos": ["n", "m15", "p15"],
        "rotations_n_trans": ["cw", "ccw"],
        "numbers": ["1", "2", "3", "4", "5"],
        "consent": ["y", "n"]
        }

    if option not in values.keys() and valid_options:  # TODO: Do as try catch clause
        values[option] = valid_options
    elif option not in values.keys() and valid_options is None:
        print("Please provide the valid inputs for your custom option")

    while True:
        print(prompt)
        print(f"Valid options: {values[option]}")
        response = str(input())

        if response in values[option]:
            break
        else:
            print("Invalid response.")

    return response


def smart_answer(user_input, options):
    """
    Function that will enable users to enter in multiple options. This function analyzes a user's input and returns
    a list of the options which were selected.
    """
    pass



if __name__ == "__main__":
    """
    Run this file like a script and you can do everything you need to here.
    """
    data_manager = AstData(my_ast_files)

    print("""
        ========= ASTERISK TEST DATA MANAGER ==========
          I MANAGE YOUR DATA FOR THE ASTERISK STUDY
              AT NO COST, STRAIGHT TO YOUR DOOR!
                           *****

        What can I help you with?
        1 - view a set of images like a video
        2 - extract a single data zip file
        3 - extract a batch of zip files
    """)
    ans = smart_input("Enter a function", "mode", ["1", "2", "3"])
    subject = smart_input("Enter subject name: ", "subjects")
    hand = smart_input("Enter name of hand: ", "hands")

    if ans == "1":
        translation = smart_input("Enter type of translation: ", "translations_w_n")

        if translation == "n":
            rotation = smart_input("Enter type of rotation: ", "rotations_n_trans")
        else:
            rotation = smart_input("Enter type of rotation: ", "rotation_combos")

        trial_num = smart_input("Enter trial number: ", "numbers")

        data_manager.view_images(subject, hand, translation, rotation, trial_num)

    elif ans == "2":
        translation = smart_input("Enter type of translation: ", "translations")
        rotation = smart_input("Enter type of rotation: ", "rotation_combos")
        trial_num = smart_input("Enter trial number: ", "numbers")

        data_manager.single_extract(subject, hand, translation, rotation, trial_num)

    elif ans == "3":
        data_manager.batch_extract(subject, hand)

    else:
        print("Invalid entry. ")
        quit()
