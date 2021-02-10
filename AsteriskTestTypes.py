import asterisk_data_manager as datamanager


class AsteriskTestTypes:
    test_type_name = ["Translation", "Rotation",
                      "Twist_translation", "undefined"]
    translation_name = ["a", "b", "c", "d", "e", "f", "g", "h", "n"]
    rotation_name = ["cw", "ccw", "n"]
    twist_name = ["p15", "m15", "n"]
    translation_angles = range(90, 90-360, -45)
    twist_directions = {"Clockwise": -15, "Counterclockwise": 15}
    rotation_directions = {"Clockwise": -25, "Counterclockwise": 25}
    # TODO: Need to include subject number

    def __init__(self):
        self.test_type_index = 3
        self.translation_index = 8
        self.rotation_index = 2

    def __str__(self):
        """Print results"""
        ret_str = f"Test: {self.test_type_name[self.test_type_index]} "

        if self.is_translation_test() or self.is_twist_translation_test():
            ret_str = ret_str + \
                f"Trans {self.get_translation_name()} {self.get_translation_angle()}"

        if self.is_rotation_test():
            ret_str = ret_str + f"Rot {self.get_rotation_name()}"

        if self.is_twist_translation_test():
            ret_str = ret_str + f" Twist {self.get_twist_name()}"

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


# ------------------------------------
def generate_fname(subject_name, hand):
    """Create the full pathname
    :param folder_path Directory where data is located -> currently not used
    :param subject_name Name of subject
    :param hand Name of hand"""

    for s, h, t, r, n in datamanager.generate_names_with_s_h(subject_name, hand):
        file_name = f"{s}_{h}_{t}_{r}_{n}.csv"

        # total_path = folder_path + file_name
        # yield total_path
        yield file_name
