#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt

from pathlib import Path
from zipfile import ZipFile


class AstData:

    def __init__(self):
        """
        Class which contains helper functions for data wrangling - getting ready for asterisk data analysis
        home - home directory of git repo
        """
        self.home = Path(__file__).parent.absolute()

    def view_images(self, subject_name, hand_name, translation_name, rotation_name, trial_number):
        os.chdir(self.home)

        data_name = f"{subject_name}_{hand_name}_{translation_name}_{rotation_name}_{trial_number}"

        file_dir = f"viz/{data_name}/"
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

        repeat = smart_input("Show again? [y/n]", "consent")
        if repeat == "y":
            # run again
            self.view_images(subject_name, hand_name, translation_name,
                             rotation_name, trial_number)
        else:
            # stop running
            quit()
    
    def single_extract(self, subject_name, hand_name, translation_name, rotation_name, trial_number):
        """
        Extract a single zip file.
        """
        folders = f"asterisk_test_data/{subject_name}/{hand_name}/"
        file_name = f"{subject_name}_{hand_name}_{translation_name}_{rotation_name}_{trial_number}"

        extract_from = folders+file_name+".zip"

        extract_to = f"viz/{file_name}"

        with ZipFile(extract_from, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Completed Extraction: {extract_to}")

    def batch_extract(self, subject_name, hand_name):
        """
        Extract a batch of zip files for a specific subject and hand
        """
        translations = ["a", "b", "c", "d", "e", "f", "g", "h", "n"]
        n_trans_rot_opts = ["cw", "ccw"]
        rotations = ["n", "p15", "m15"]
        num = ["1", "2", "3"]  # , "4", "5"] #for now, since missing random trials 4 and 5 across the study

        for t in translations:
            if t == "n":  # necessary to divide rotations because cw and ccw only happen with no translation
                if hand_name in ["basic", "m2stiff", "modelvf"]:
                    continue
                else:
                    rot = n_trans_rot_opts
            else:
                if hand_name in ["basic", "m2stiff", "modelvf"]:
                    rot = "n"
                else:
                    rot = rotations

            for r in rot:
                for n in num:
                    self.single_extract(subject_name, hand_name, t, r, n)


def smart_input(prompt, option, valid_options=None):
    """
    Asks for input and continues asking until there is a valid response
    """
    values = {
        "subjects": ["sub1", "sub2", "sub3"],
        "hands": ["2v2", "2v3", "3v3", "barrett", "basic", "human", "m2active", "m2stiff", "modelvf"],
        "translations": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "rotations": ["n", "m15", "p15"],
        "rotations_n_trans": ["cw", "ccw"],
        "numbers": ["1", "2", "3", "4", "5"],
        "consent": ["y", "n"]
        }

    print(option)
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


if __name__ == "__main__":
    """
    Run this file like a script and you can do everything you need to here.
    """
    data_manager = AstData()

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
        translation = smart_input("Enter type of translation: ", "translations")
        rotation = smart_input("Enter type of rotation: ", "rotations")
        trial_num = smart_input("Enter trial number: ", "numbers")

        data_manager.view_images(subject, hand, translation, rotation, trial_num)

    elif ans == "2":
        translation = smart_input("Enter type of translation: ", "translations")
        rotation = smart_input("Enter type of rotation: ", "rotations")
        trial_num = smart_input("Enter trial number: ", "numbers")

        data_manager.single_extract(subject, hand, translation, rotation, trial_num)

    elif ans == "3":
        data_manager.batch_extract(subject, hand)

    else:
        print("Invalid entry. Please try again.")
        quit()
