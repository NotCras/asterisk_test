


class AstIterator:

    def __init__(self, option_dict):
        """Class which handles iterator functions. 

        Args:
            option_dict (dict{string:list(string)}): key (string) is a label for an option set. The value is a list of strings that make up that value set.

            You must pass in a dictionary with the following keys:
            hands -> all hands you wish to include in labeling
            hands_cant_rotate -> hands included in hands but are unable to do rotation trials
            subjects -> all subject names
            translations_all -> all options for translation (including "n" for rotation-only trials)
            translations -> all translation directions (excluding "n" for rotation-only trials)
            tr_r_conditions -> all rotation conditions that pair with translations
            rotation_only -> rotation trial labels for trials with no translation (a translation label of "n") 
            numbers -> trial numbers (list of strings of trial numbers, e.g. ["1", "2", ...])
        """
        self.options = option_dict    
        self.options["consent"] = ["y", "n"] # for smart_input

    def get_option_list(self, key):
        """Returns list of options for a particular key

        Args:
            key (_type_): _description_

        Returns:
            _type_: _description_

        """
        return self.options[key]

    def generate_labels_tr_pairs(self, hand_name, include_tr_trials=False, include_rotation_only_trials=True, rotation_centric=True):
        """Generator that yields all t+o combinations. 
        Will yield all combinations for a specific t value all together

        Args:
            hand_name (string): string of hand name, that must be in list of hands option
            include_tr_trials (bool, optional): Include t+r combinations. Defaults to False.
            include_rotation_only_trials (bool, optional): include r+o combinations. Defaults to True.
            rotation_centric (bool, optional): yield rotation conditions around 

        Yields:
            t: translation direction label
            r: rotation label
        """
        hand_cant_do_rotation = hand_name in self.get_option_list("hands_cant_rotate")

        if rotation_centric:
            self._generate_tr_rotation_centric(hand_cant_do_rotation, include_tr_trials, include_rotation_only_trials)
        else: 
            # translation centric, that means that all translations for a specific rotation condition will be output together
            self._generate_tr_translation_centric(hand_cant_do_rotation, include_tr_trials, include_rotation_only_trials)

    def _generate_tr_translation_centric(self, hand_cant_do_rotation, include_tr_trials, include_rotation_only_trials):

        translation_directions = self.get_option_list("translations_all")
        rotation_only_directions = self.get_option_list("rotation_only")
        tr_rotation_conditions = self.get_option_list("tr_r_conditions")

        for t in translations:
            # include or exclude rotation-only trials
            if t == "n" and include_rotation_only_trials:
                # we also need to check if the hand can do rotations
                if hand_cant_do_rotation:
                    continue
                else:
                    rot = rotation_only_directions

            elif t == "n" and not include_rotation_only_trials:
                # if we do not include rotation-only, skip this trial
                continue

            else:
                # otherwise, if we have a translation direction, then we yield the direction with rotation conditions
                if hand_cant_do_rotation or not include_tr_trials:
                    rot = ["n"]

                else:
                    rot = tr_rotation_conditions
            
            for r in rot:
                yield t,r

    def _generate_tr_rotation_centric(self, hand_cant_do_rotation, include_tr_trials, include_rotation_only_trials):
        
        translation_directions = self.get_option_list("translations_all")
        rotation_only_directions = self.get_option_list("rotation_only")
        tr_rotation_conditions = self.get_option_list("tr_r_conditions")

        rotation_options = []
        if include_tr_trials:
            rotation_options.append(tr_rotation_conditions)
        
        if include_rotation_only_trials:
            rotation_options.append(rotation_only_directions)

        for r in rotation_options:
            for t in translations:
            
                if r in ["m15", "p15"] and include_tr_trials:
                    # skip extra rotation conditions if the hand can't do rotations
                    if hand_cant_do_rotation:
                        continue

                    # tr rotation conditions do not pair with "n"
                    if t == "n":
                        continue

                elif r in ["cw", "ccw"] and include_rotation_only_trials:
                    # skip rotation-only trials if the hand can't do rotations
                    if hand_cant_do_rotation:
                        continue

                    # rotation-only trial labels should only be paired with a t of "n"
                    if t != "n":
                        continue                

                yield t, r

    def generate_labels_htrsn_sets(self, hand_name, subject_name, include_tr_trials=False, include_rotation_only_trials=True, rotation_centric=True):
        """Generates all trial combinations with a specific hand name and subject

        Args:
            subject_name (_type_): _description_
            hand_name (_type_): _description_
            include_tr_trials (bool, optional): _description_. Defaults to False.
            include_rotation_only_trials (bool, optional): _description_. Defaults to True.
            rotation_centric (bool, optional): _description_. Defaults to True.

        Yields:
            h: hand label
            t: translation direction label
            r: rotation label
            s: subject name
        """
        num_of_trials = self.get_option_list("numbers")

        for t, r in self.generate_labels_tr_pairs(hand_name, include_tr_trials=include_tr_trials, include_rotation_only_trials=include_rotation_only_trials, rotation_centric=rotation_centric):
            for n in num:
                yield hand_name, t, r, subject_name, n

    def generate_labels_all(self, hand_names=None, subjects=None, include_tr_trials=False, include_rotation_only_trials=True, rotation_centric=True):
        """_summary_

        Args:
            hand_names (_type_, optional): _description_. Defaults to None.
            subjects (_type_, optional): _description_. Defaults to None.
            include_tr_trials (bool, optional): _description_. Defaults to False.
            include_rotation_only_trials (bool, optional): _description_. Defaults to True.
            rotation_centric (bool, optional): _description_. Defaults to True.
        """
        if subjects is None:
            subjects = self.get_option_list("subjects")
        
        if hand_names is None:
            hand_names = self.get_option_list("hands")

        for h in hand_names:
            for s in subjects:
                yield self.generate_labels_htrsn_sets(h, s)

    def smart_input(prompt, option, valid_options=None):
        """_summary_

        Args:
            prompt (_type_): _description_
            option (_type_): _description_
            valid_options (_type_, optional): _description_. Defaults to None.
        """
        if option not in values.keys() and valid_options:  # TODO: Do as try catch clause
            values[option] = valid_options

        elif option not in values.keys() and valid_options is None:
            print("Please provide the valid inputs for your custom option")

        prompt_attempts = 0

        print(prompt)
        print(f"Valid options: {values[option]}")
        while prompt_attempts < 4:

            response = str(input())

            if response in values[option]:
                break
            else:
                prompt_attempts += 1
                print("Invalid response.")

        return response


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


def smart_answer(user_input, options):
    """
    Function that will enable users to enter in multiple options. This function analyzes a user's input and returns
    a list of the options which were selected.
    """
    pass

my_option_dict = {
    hands: ["2v2", "2v3", "3v3", "basic", "m2active", "palm1r", "palm2r"],
    hands_cant_rotate: ["basic", "m2active", "palm1r"],
    subjects: ["sub1", "sub2", "sub3"],
    translations_all: ["a", "b", "c", "d", "e", "f", "g", "h", "n"],
    translations: ["a", "b", "c", "d", "e", "f", "g", "h"],
    tr_r_conditions: ["n", "m15", "p15"],
    rotation_only: ["cw", "ccw"],
    numbers: ["1", "2", "3", "4", "5"]
}
