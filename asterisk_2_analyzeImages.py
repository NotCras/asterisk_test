#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kartik (original), john (edits, cleaning)
"""
import numpy as np
import sys
if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2, PIL
from cv2 import aruco
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
#%matplotlib nbagg
import os
import time
import asterisk_0_prompts as prompts
import asterisk_0_dataHelper as helper


## === IMPORTANT ATTRIBUTES ===
marker_side = 0.03
processing_freq = 1 #analyze every 1 image

## ============================

def inversePerspective(rvec, tvec):
    #print(rvec)
    #print(np.matrix(rvec[0]).T) #TODO: Something wrong is going on here... need to look into the rodrigues function
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = np.expand_dims(rvec1.squeeze(),1), np.expand_dims(tvec1.squeeze(),1)
    rvec2, tvec2 = np.expand_dims(rvec2.squeeze(),1), np.expand_dims(tvec2.squeeze(),1)
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))

    return composedRvec, composedTvec

def estimatePose(frame, marker_side, mtx,dist):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250) # MAKE SURE YOU HAVE RIGHT ONE!!!!
    # detector parameters can be set here (List of detection parameters[3])
    parameters =  aruco.DetectorParameters_create()
    #parameters.adaptiveThreshConstant = 10

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    #print(f"corners: {corners}, ids: {ids}")

    if np.all(ids != None):
        #print("Found a tag.")
        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_side, mtx, dist) #TODO: quickfix is corners[0]... is there a more elegant way to fix?

    else:
        print("Could not find marker in frame.")
        #quit()

    return rvec, tvec, corners

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def pose_estimation_process(folder, image_tag, mtx_val, dist_val, init_corners, init_rvec, init_tvec ):
    frame = cv2.imread(os.path.join(folder, image_tag))

    next_rvec, next_tvec, next_corners = estimatePose(frame,marker_side, mtx_val, dist_val)
    next_corners = next_corners[0].squeeze()

    #print(f"calculating angle, {next_corners}")
    rel_angle  = angle_between(init_corners[0]-init_corners[2],next_corners[0]-next_corners[2])
    rel_rvec, rel_tvec = relativePosition(init_rvec, init_tvec, next_rvec, next_tvec)

    translation_val = np.round(np.linalg.norm(rel_tvec),4)
    rotation_val = rel_angle*180/np.pi

    rotM = np.zeros(shape=(3,3))
    cv2.Rodrigues(rel_rvec, rotM, jacobian = 0)
    ypr = cv2.RQDecomp3x3(rotM)

    return rel_rvec, rel_tvec, translation_val, rotation_val, ypr

#mtx = camera intrinsic matrix , dist =  distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])

#================================================================
def analyze_images(data_path, subject_name, hand, dir_label, trial_type, trial_num):
    frame = cv2.imread(os.path.join(data_path,'left0000.jpg'))
    orig_rvec, orig_tvec, orig_corners = estimatePose(frame,marker_side, mtx, dist)
    orig_corners = orig_corners[0].squeeze()
    #print("Tag found in initial image.")

    total = 0

    f = []
    counter = 0
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        f.extend(filenames)
        f.sort()
        break

    #print(f)
    #print(" ")
    #print(dirpath)
    #print(" ")

    while(True):

        for image_ in f:
            #print(image_)
            if '.ini' in image_:
                # print("Configuration file found. Skipping over.")
                # camera configuration file, skip over
                continue

            if np.mod(counter, processing_freq) > 0:
                continue

            try:
                rel_rvec, rel_tvec, translation, rotation, ypr = pose_estimation_process(data_path, image_, mtx, dist, orig_corners, orig_rvec, orig_tvec)
                #print(f"Succeeded at image {counter}")
                
            except Exception as e: 
                print(f"Error with finding ARuco tag in image {counter}.")
                print(e)
                counter += 1
                continue

            counter += 1
            total +=1

            rel_pose = np.concatenate((rel_rvec,rel_tvec))

            data_file = subject_name + "_" + hand + "_" + dir_label + "_" + trial_type + "_" + trial_num + ".csv"
            csv_loc = "csv/" + data_file

            with open(csv_loc,'a') as fd:
                for i in rel_pose:
#                       for y in i:
                    fd.write(str(i[0]))
                    fd.write(',')
                    # print('here')

                fd.write(str(translation))
                fd.write(',')
                fd.write(str(rotation))
                fd.write('\n')
#                    print(rel_pose)

            #print('Total: ' + str(total) +' Done '+ image_)

        print("          ")
        print('Completed ' + data_file)
        print("Finished: " + str(total) + "/" + str(counter) )
        break



if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    mtx = np.array(((617.0026849655,-0.153855356,315.5900337131),#fx, s,cx
                   (0,614.4461785395,243.0005874753), ##0,fy,cy
                   (0,0,1) ))
    dist = np.array((0.1611730644,-0.3392379107,0.0010744837,	0.000905697)) #k1,k2,p1,p2 ie radial dist and tangential dist

#================ FILE PATH TO IMAGE FOLDER =====================

    subject_name = helper.collect_prompt_data(
        prompts.subject_name_prompt, prompts.subject_name_options)

    hand = helper.collect_prompt_data(
        prompts.hand_prompt, prompts.hand_options)

    folder_path = "asterisk_test_data/" + subject_name + "/" + hand

    files_covered = list()

    print("FOLDER PATH")
    print(folder_path)

    #construct all the zip file names

    if hand == "basic" or hand == "m2stiff" or hand == "modelvf":
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

                inner_path = "viz/" + subject_name + "_" + hand + "_" + d + "_" + t + "_" + num + "/"
                os.chdir(home_directory)
                #data_path = inner_path
                print(inner_path)

                try:
                    analyze_images(inner_path, subject_name, hand, d, t, num)
                    files_covered.append(zip_file)
                except Exception as e: 
                    print(e)
                    files_covered.append(f"FAILED: {zip_file}")

    print("DONE!!")
    print(files_covered)
    


