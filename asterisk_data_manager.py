#!/usr/bin/env python3

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from zipfile import ZipFile

class ast_data():

    def __init__(self):
        '''
        Class which contains helper functions for data wrangling - getting ready for asterisk data analysis
        home - home directory of git repo
        '''
        self.home = Path(__file__).parent.absolute()

    def view_images(self, subject_name, hand, dir_label, trial_type, trial_num):
        os.chdir(self.home)

        data_name = subject_name + "_" + hand + "_" + \
            dir_label + "_" + trial_type + "_" + trial_num

        file_dir = "viz/" + data_name + "/"
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

        repeat = input("Show again? [y/n]")
        if(repeat == "y"):
            #run again
            self.view_images(subject_name, hand, dir_label,
                             trial_type, trial_num)
        else:
            #stop running
            #quit()
            pass
    
    def single_extract(self):
        '''
        Extract a single zip file.
        '''
        pass

    def batch_extract(self):
        '''
        Extract a batch of zip files.
        '''
        pass

def yield_names():
    #subject, hand, type_trans, type_rot,

if __name__ == "__main__":
    '''
    Run this file like a script and you can do everything you need to here.
    '''
    data_manager = ast_data()

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
    ans = input("Enter a function")

    if(ans == 1):
        #TODO: add prompt here

        #data_manager.view_images()
        pass

    elif(ans == 2):
        #TODO: add prompt here

        #data_manager.single_extract()
        pass

    elif(ans == 3):
        #TODO: add prompt here

        #data_manager.batch_extract()
        pass

    else:
        print("Invalid entry. Please try again.")
        quit()


