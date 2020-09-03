# So you want to run the asterisk test?
Resources for running the asterisk test.


### initial setup
This uses python 3. 

Dependencies to run all of the scripts are:
- opencv (with aruco package)
- curtsies
- keyboard

Install with pip.

---

### step 0 - data helper
This is a script to help us collect camera images and save them in the right folder (without having the test proctor worry about it). 

Will save all data, with correct folder structure, in a folder called `data/`. Will also generate a zip file of the data in the folder root. We back up those zip files to box.

We made a folder called `zips/` where we stored these zip files as an extra backup on the computer at the test location.

I made the prompts file to help clean up the data helper file.

---


### step 1 - data extraction
I made this step to extract the zips that I downloaded from box. This script is wired to grab the zip files in `asterisk_test_folder/[sub_name]/[hand]`. 

It will extract the files to a folder named `viz/`. Inside are folders named for each trial.


---

### step 2 - analyzing images
Contains the code to analyze each image for the aruco tag. All aruco tag locations are relative to the aruco tag location in the initial frame. 

Generates csv file in `csv/` folder. (I know, lots of folders but I want the data for every step just in case something goes wrong)

**csv columns**: roll, pitch, yaw, x, y, z, magnitude of rotation from initial pose, magnitude of translation from initial pose

---

### step 3 - data conditioning

`[NOT DONE]` 

Make data nice for frechet distance analysis. 

---


### step 4 - generate companion path

`[NOT DONE]` 

Generate ground truth path for frechet distance analysis.

---

### step 5 - frechet distance

`[NOT DONE]` 



