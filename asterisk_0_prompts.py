#------------------------------------
subject_name_prompt = """
ENTER SUBJECTS NAME
(lowercase!)

Possible options:
"""
subject_name_options = ["john", "josh", "sage", "garth", "test"]

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
def request_name_hand_simple(path):
    #import csv file of data
    #for now, just write it out each time
    folder = path #TODO: make smarter
    sub = helper.collect_prompt_data(
        subject_name_prompt, subject_name_options)
    h = helper.collect_prompt_data(
        hand_prompt, hand_options)
    #t = "none"
    #d = "a"

    return folder, sub, h

#------------------------------------
def request_name_hand_num_simple(path):
    #import csv file of data
    #for now, just write it out each time
    folder = path #TODO: make smarter
    sub = helper.collect_prompt_data(
        subject_name_prompt, subject_name_options)
    h = helper.collect_prompt_data(
        hand_prompt, hand_options)
    #t = "none"
    #d = "a"
    num = helper.collect_prompt_data(
        trial_prompt, trial_options)

    return folder, sub, h, num

#------------------------------------
class AsteriskTestTypes:
    test_type_name = ["Translation", "Rotation", "Twist_translation", "undefined"]
    translation_name = ["a", "b", "c", "d", "e", "f", "g", "h", "none"]
    rotation_name = ["cw", "ccw", "none"]
    twist_name = ["plus15", "minus15", "none"]
    translation_angles = range(90, 90-360, -45)
    twist_directions = {"Clockwise": -15, "Counterclockwise": 15}
    rotation_directions = {"Clockwise": -25, "Counterclockwise": 25}

    def __init__(self):
        self.test_type_index = 3
        self.translation_index = 8
        self.rotation_index = 2

    def __str__(self):
        """Print results"""
        ret_str = "Test: {0} ".format(self.test_type_name[self.test_type_index])
        if self.is_translation_test() or self.is_twist_translation_test():
            ret_str = ret_str + "Trans {0} {1}".format(self.get_translation_name(), self.get_translation_angle())
        if self.is_rotation_test():
            ret_str = ret_str + "Rot {0}".format(self.get_rotation_name())
        if self.is_twist_translation_test():
            ret_str = ret_str + " Twist {0}".format(self.get_twist_name())
        return ret_str

    def set(self, att):
        self.test_type_index = att.test_type_index
        self.translation_index = att.translation_index
        self.rotation_index = att.rotation_index

    def set_translation_test(self, in_translation_index: int):
        self.test_type_index = 0
        self.translation_index = in_translation_index
        self.rotation_index = 2

    def set_rotation_test(self, in_rotation_index: int):
        self.test_type_index = 1
        self.translation_index = 8
        self.rotation_index = in_rotation_index

    def set_twist_translation_test(self, in_translation_index: int, in_rotation_index: int):
        self.test_type_index = 2
        self.translation_index = in_translation_index
        self.rotation_index = in_rotation_index

    def is_translation_test(self) -> bool:
        """Is this a translation test?
        :returns true/false"""
        if self.test_type_index == 0:
            return True
        return False

    def is_rotation_test(self) -> bool:
        """Is this a rotation test?
        :returns true/false"""
        if self.test_type_index == 1:
            return True
        return False

    def is_twist_translation_test(self) -> bool:
        """Is this a twist then translation test?
        :returns true/false"""
        if self.test_type_index == 2:
            return True
        return False

    def get_test_name(self) -> str:
        return self.test_type_name[self.test_type_index]

    def get_translation_name(self) -> str:
        return self.translation_name[self.translation_index]

    def get_translation_angle(self) -> int:
        return self.translation_angles[self.translation_index]

    def get_rotation_name(self) -> str:
        return self.rotation_name[self.rotation_index]

    def get_twist_name(self) -> str:
        return self.twist_name[self.rotation_index]

    def get_file_name(self, trial=-1) -> str:
        """ The file name
         :param trial is which trial (if any)
         returns: String, no Trial appended if trial is -1"""
        trial_str = ""
        if trial is not -1:
            trial_str = "_{0}".format(trial)

        if self.is_translation_test():
            return self.get_translation_name() + "_none" + trial_str
        if self.is_rotation_test():
            return self.get_rotation_name() + "_none" + trial_str
        if self.is_twist_translation_test():
            return self.get_twist_name() + "_" + self.get_translation_name() + trial_str
        return "NotValid"

    def is_type(self, in_att) -> bool:
        """ See if this is the same type
        :param in_att another instance of AsteriskTestType
        :returns true/false"""
        if in_att.test_type_index is not self.test_type_index:
            return False
        if in_att.translation_index is not self.translation_index:
            return False
        if in_att.rotation_index is not self.rotation_index:
            return False
        return True

    @staticmethod
    def generate_translation():
        for i in range(0, 8):
            att = AsteriskTestTypes()
            att.set_translation_test(i)
            yield att

    @staticmethod
    def generate_rotation():
        for i in range(0, 2):
            att = AsteriskTestTypes()
            att.set_rotation_test(i)
            yield att

    @staticmethod
    def generate_twist_translation():
        for ir in range(0, 2):
            for it in range(0, 8):
                att = AsteriskTestTypes()
                att.set_twist_translation_test(it, ir)
                yield att


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

