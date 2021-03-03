# So you want to run the asterisk test?
In this git repository you will find a collection of tools and resources for running a complete asterisk test.
from start to finish.


### Initial Setup
We use python 3. We have the following dependencies:
- *matplotlib*
- *pandas*
- *numpy*
- *similaritymeasures*
- *opencv** ( opencv-contrib-python )
- *curtsies**
- *keyboard**

Install with pip. 

**Note:* You can ignore *curtsies* and *keyboard* if you will not use the ***asterisk_trial_helper.py*** script. 
Similarly, you can ignore *opencv* if you will not use the ***asterisk_aruco.py*** script.

---
## Folder Setup
The asterisk test adheres to a strict folder structure. We provide this as part of the git repository.
- *asterisk_test_data* -> folder for storing zip files of the study
- *viz* -> this folder holds the image sets that correspond to each study.
- *csv* -> folder to store aruco positions gathered from viz data as csv files
- *documentation* -> self explanatory. For documentation we use pdoc.
- *filtered* -> filtered data is stored here. 
- *resources* -> extra resources for running the asterisk test such as aruco codes, a webcam streaming script,
and brief instructions on the asterisk test protocol

*optional*
- data -> 
- pics -> 

---
## Running the Demo
The best way to understand each file is to check the documentation and to try the demo. We have provided
a demo which includes an example workflow using the asterisk test suite. Your starting place for
this is the ***asterisk_main.py*** file.