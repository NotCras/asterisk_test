# So you want to run the asterisk test?
In this git repository you will find a collection of tools and 
resources for running a complete asterisk test from start to finish.

This started out as helper code to help run the Asterisk Test, however 
it continued to take on mass and incorportated analysis code.
Our intent with this library is to provide code for the Asterisk Test
that will help with running your own test, exploring the test data, 
and simplify basic analysis and plotting.

For a concise description of how to use the asterisk test data, please
start at asterisk_test_demo.py. We used pdoc for documentation, and you
can find documentation (which we will try to keep up to date) in the 
documentation folder.

### Initial Setup
We use python 3. We have the following dependencies:
- *matplotlib*
- *pandas*
- *numpy*
- *similaritymeasures*
- *opencv** ( opencv-contrib-python )
- *curtsies**
- *keyboard**

Install dependencies with pip.

Currently, we are considering configuring this package for pip, but we will need to find a 
suitable solution for managing folders first. For now, install the package by cloning it.

**Note:* You can ignore *curtsies* and *keyboard* if you will not use the ***trial_helper.py*** script. 
Similarly, you can ignore *opencv* if you will not use the ***ast_aruco.py*** script.

---
## Folder Setup
The asterisk test adheres to a strict folder structure. We provide this as part of the git repository.
- *compressed_data* -> folder for storing zip files of your study data
- *viz* -> this folder holds the image sets that correspond to each study
- *aruco_data* -> folder to store aruco positions gathered from viz data as csv files
- *documentation* -> self explanatory. For documentation we use pdoc.
- *trial_paths* -> self explanatory. Saved data (filtered or not) is stored here. 
- *resources* -> extra resources for running the asterisk test such as aruco codes, a webcam streaming script,
and brief instructions on the asterisk test protocol
  
- *results* -> stores metric data here. Has a folder for storing saved images.

---
## More Basic Details

Will be completed later.