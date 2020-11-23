#------------------------------------
subject_name_prompt = """
ENTER SUBJECTS NAME
(lowercase!)

Possible options:
"""
subject_name_options = ["john", "josh", "sage", "garth"]

#------------------------------------
hand_prompt = """
ENTER HAND YOU ARE USING FOR THIS TRIAL
(lowercase!)

Possible options:
"""
hand_options = ["human", "basic",  "m2stiff", "m2active",
                "2v2", "3v3", "2v3", "barrett", "modelvf"]  # "modelo", "modelk",

#------------------------------------
dir_prompt = """
ENTER DIRECTION OF CURRENT TRIAL
(lowercase!)

Possible options:
"""
dir_options = ["a", "b", "c", "d", "e", "f", "g", "h", "cw", "ccw"]
dir_options_no_rot = ["a", "b", "c", "d", "e", "f", "g", "h"]

#------------------------------------
trial_prompt = """
WHAT NUMBER TRIAL IS THIS
(lowercase! ... :P)

Up to ...
"""
trial_options = ["1", "2", "3", "4", "5"]  # , "6", "7", "8", "9", "10"]

#------------------------------------
type_prompt = """
WHAT TYPE OF TRIAL IS THIS
(lowercase!)

Options ...
"""
type_options = ["none", "plus15", "minus15"]

#------------------------------------
check_prompt = "Are you happy with this data? : "
check_options = ["yes", "no", "cancel"]

temp_file_check = "Are you still doing"

#------------------------------------
def generate_fname(folder_path, subject_name, hand):
    """Create the full pathname
    :param folder_path Directory where data is located
    :param subject_name Name of subject
    :param hand Name of hand"""

    if hand == "basic" or hand == "m2stiff" or hand == "vf":
        types = ["none"]
    else:
        types = type_options

    for t in types:
        if t == "none":
            directions = dir_options
        else:
            directions = dir_options_no_rot

        for d in directions:
            for num in ["1", "2", "3"]:
                file_name = subject_name + "_" + hand + "_" + d + "_" + t + "_" + num + ".csv"

                total_path = folder_path + file_name
                yield total_path
