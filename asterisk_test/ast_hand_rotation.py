import pdb

import matplotlib.pyplot as plt
import numpy as np
from ast_hand_translation import AstHandTranslation
from ast_avg_rotation import AveragedRotation
from matplotlib.patches import Wedge
import data_manager as datamanager
import ast_trial as atrial
import ast_trial_rotation as arot


class AstHandRotation(AstHandTranslation):
    def __init__(self, subjects, hand_name):
        super().__init__(subjects, hand_name, rotation=None)
        # TODO: is it ok to just leave set_rotation var hanging around?

    def _gather_hand_data(self, subjects, blocklist=None, normalized_data=True):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        :param subjects: list of subjects to get
        """
        data_dictionary = dict()

        for r in datamanager.generate_options("rotations_n_trans"):
            key = f"n_{r}"
            data = self._make_asterisk_trials_from_filenames(subjects, r,
                                                             datamanager.generate_options("numbers"),
                                                             blocklist=blocklist, normalized_data=normalized_data)
            if data:
                data_dictionary[key] = data
                # pdb.set_trace()
            else:
                print(f"{key} not included, no valid data")
                # pdb.set_trace()

        return data_dictionary

    def _make_asterisk_trials_from_filenames(self, subjects, rotation_label, trials,
                                             blocklist=None, normalized_data=True):
        """
        Goes through data and compiles data with set attributes into an AsteriskTrial objects
        :param subjects: name of subject
        :param translation_label: name of translation trials
        :param rotation_label: name of rotation trials
        :param trial_num: trial numbers to include, default parameter
        """
        # Maybe make it return None? then we can return all dictionary keys that don't return none in the other func

        gathered_data = list()
        for s in subjects:  # TODO: subjects is a list, make a type recommendation?
            for n in trials:
                asterisk_trial = f"{s}_{self.hand.get_name()}_n_{rotation_label}_{n}"

                if blocklist is not None and asterisk_trial in blocklist:
                    print(f"{asterisk_trial} is blocklisted and will not be used.")
                    continue

                try:
                    trial_data = arot.AstTrialRotation(f"{asterisk_trial}.csv", norm_data=normalized_data)
                    print(f"{trial_data.generate_name()}, labels: {trial_data.path_labels}")

                    gathered_data.append(trial_data)

                except Exception as e:
                    print(f"AstTrial generation failed for {asterisk_trial}")
                    print(e)
                    #print(" ")
                    continue

        return gathered_data

    def _get_directions_in_data(self):
        """
        Returns a list of trial directions that exist in the data
        :return:
        """
        list_of_dirs = list()
        for k in list(self.data.keys()):
            t, r = k.split("_")
            if t != "n":
                continue
                # TODO: should we throw an error here?
            list_of_dirs.append(r)

        return list_of_dirs

    def _get_ast_set(self, subjects, trial_number=None, exclude_path_labels=None):
        """
        Picks out an asterisk of data (all translational directions) with specific parameters
        :param subjects: specify the subject or subjects you want
        :param trial_number: specify the number trial you want, if None then it will
            return all trials for a specific subject
        :param rotation_type: rotation type of batch. Defaults to "n"
        """
        dfs = []
        rotations = datamanager.generate_options("rotations_n_trans")  # ["cw", "ccw"]

        for direction in rotations:
            dict_key = f"n_{direction}"
            # TODO: maybe we set the rotation type per hand trial... might be easier to handle
            trials = self.data[dict_key]
            # print(f"For {subject_to_run} and {trial_number}: {direction}")

            for t in trials:
                # print(t.generate_name())
                if trial_number:  # if we want a specific trial, look for it
                    if (t.subject == subjects) and (t.trial_num == trial_number):
                        for l in t.path_labels:
                            if exclude_path_labels is not None and l in exclude_path_labels:
                                continue  # skip trial if it has that path_label

                        dfs.append(t)
                    elif (t.subject in subjects) and (t.trial_num == trial_number):
                        for l in t.path_labels:
                            if exclude_path_labels is not None and l in exclude_path_labels:
                                continue  # skip trial if it has that path_label

                        dfs.append(t)

                else:  # otherwise, grab trial as long as it has the right subject
                    if t.subject == subjects or t.subject in subjects:
                        for l in t.path_labels:
                            if exclude_path_labels is not None and l in exclude_path_labels:
                                continue  # skip trial if it has that path_label

                        dfs.append(t)

        return dfs

    def _get_ast_dir(self, direction_label, subjects, exclude_path_labels=None):
        """
        Get all of the trials for a specific direction. You can specify subject too
        :param direction_label: translation direction
        :param subjects: subject or list of subjects to include
        :param rotation_label: rotation label, defaults to "n"
        """
        dict_key = f"n_{direction_label}"
        direction_trials = self.data[dict_key]
        gotten_trials = []
        dont_include = False

        for t in direction_trials:
            if t.subject == subjects or t.subject in subjects:
                # check if trial has a path_label that we don't want to include
                for l in t.path_labels:
                    if exclude_path_labels is not None and l in exclude_path_labels:
                        dont_include = True
                        # continue  # skip trial if it has that path_label

                # if it passes path_label check, add it to the
                if dont_include:
                    dont_include = False
                else:
                    gotten_trials.append(t)

        return gotten_trials

    def _average_dir(self, direction, subject=None, exclude_path_labels=None):
        """
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param direction: trial direction to average
        :param rotation: trial rotation to average
        :param subject: subject or list of subjects to average, optional. If not provided, defaults to all subjects
        :return returns averaged path
        """

        # TODO: average the max rotation number for each trial in group, also provide std (continue) -v
        #  then also average largest magnitude of translation for each trial

        if subject is None:  # get batches of data by trial type, if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(direction_label=direction, subjects=self.subjects_containing,
                                       exclude_path_labels=exclude_path_labels)

        else:
            trials = self._get_ast_dir(direction_label=direction, subjects=subject,
                                       exclude_path_labels=exclude_path_labels)

        if trials:
            average = AveragedRotation(direction=direction, trials=trials)
            return average

        else:
            print(f"No trials for n_{direction}, skipping averaging.")
            return None

    def calc_averages(self, subjects=None, exclude_path_labels=None):
        """
        calculate and store all averages
        :param subjects: subject(s) to include in the average. Defaults to all subjects in object
        :param rotation: refers to the rotation type ("n", "m15", "p15"). Defaults to all options
        """
        averages = []
        if subjects is None:  # if no subjects given, defaults to all subjects
            subjects = self.subjects_containing

        dirs = self._get_directions_in_data()
        print(f"Directions included: {dirs}")

        for r in datamanager.generate_options("rotations_n_trans"):
            # make sure that we only include translations that are in the data
            if r in dirs:
                print(f"Averaging {r}")
                avg = self._average_dir(direction=r, subject=subjects, exclude_path_labels=exclude_path_labels)
                if avg is not None:
                    averages.append(avg)

        self.averages = averages
        return averages

    def plot_ast_avg(self, subjects=None, show_plot=True, save_plot=False, include_notes=True,
                     linestyle="solid", plot_contributions=False, exclude_path_labels=None):

        fig_size = 7
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw=dict(aspect="equal"))

        if subjects is None:
            subjects = self.subjects_containing

        cw_rot = None
        ccw_rot = None
        cw_rot_std = None
        ccw_rot_std = None
        cw_a = None
        ccw_a = None

        if self.averages:
            for a in self.averages:
                if a.direction == "cw":
                    cw_a = a
                    cw_rot = np.abs(a.max_rot[0])
                    cw_rot_std = a.max_rot[1]
                elif a.direction == "ccw":
                    ccw_a = a
                    ccw_rot = np.abs(a.max_rot[0])
                    ccw_rot_std = a.max_rot[1]
                else:
                    # TODO: throw an error here?
                    print("We have an average that doesn't belong!")

            if cw_rot is None or ccw_rot is None:
                print(f"Error with existing averages. cw:{cw_rot}, ccw:{ccw_rot}")

        else:
            cw_a = self._average_dir("cw", subject=subjects)
            cw_rot = np.abs(cw_a.max_rot[0])
            cw_rot_std = cw_a.max_rot[1]

            ccw_a = self._average_dir("ccw", subject=subjects)
            ccw_rot = np.abs(ccw_a.max_rot[0])
            ccw_rot_std = ccw_a.max_rot[1]

        print(f"Values!!! cw: {cw_rot}, ccw: {ccw_rot}")
        data = [cw_rot, ccw_rot, 360 - cw_rot - ccw_rot]  # TODO: is there some way to add standard deviations?

        labels = [f"{str(int(cw_rot))}{chr(176)}{chr(177)}{str(int(cw_rot_std))}",
                  f"{str(int(ccw_rot))}{chr(176)}{chr(177)}{str(int(ccw_rot_std))}", ""]  # "not rotated"]

        colors = ["crimson", "royalblue", "whitesmoke"]
        # colors = [(244./255., 178./255., 191./255.), (193./255., 205./255., 245./255.), "whitesmoke"]

        ax.pie(data, colors=colors, labels=labels, labeldistance=1.05, wedgeprops=dict(width=0.2, alpha=1.0),
               startangle=90 - cw_rot, counterclock=True,
               textprops=dict(color="darkslategrey", size=12, weight="bold",
               # rotation_mode = 'anchor', va='center', ha='center'
               ))

        # # draw circle by drawing a white circle on top of center
        # centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        # fig = plt.gcf()
        # fig.gca().add_artist(centre_circle)

        # draw standard deviations
        def pie_std(ang, std, color):
            test_line = Wedge((0, 0), 1, ang - 0.05, ang + 0.05, width=0.2, color="black", alpha=1)
            test1 = Wedge((0, 0), 1, ang - std, ang, width=0.2, color=color, alpha=1)
            test2 = Wedge((0, 0), 1, ang, ang + std, width=0.2, color=color, alpha=1)

            fig = plt.gcf()
            fig.gca().add_artist(test1)
            fig.gca().add_artist(test2)
            fig.gca().add_artist(test_line) # has to go last, otherwise won't show up

        # light crimson -> Cupid #F4B2BF, RGB = 244	178	191
        pie_std(90-cw_rot, cw_rot_std, (244./255., 178./255., 191./255.))  # 90 - cw_rot is there because of 90 deg rotation of pie from origin
        # light royal blue -> Periwinkle #C1CDF5, RGB = 193	205	245
        pie_std(90+ccw_rot, ccw_rot_std, (193./255., 205./255., 245./255.)) # 90 + ccw_rot same reason as line above

        if include_notes:
            self._plot_notes([cw_a, ccw_a], ax=ax)

        self._plot_translations(cw_a, ccw_a)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        # plt.tight_layout()

        fig.suptitle(f"Avg {self.hand.get_name()}, Standing Rotations", fontweight='bold')
        ax.set_title("Cube size: ~0.25 span, init pos: 0.75 depth")  #, pad=10)

        if save_plot:
            plt.savefig(f"results/pics/avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_standing_rotations.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def _plot_translations(self, cw_a, ccw_a):
        # line_x = [0, 0.1, 0.2, 0.3]
        # line_y = [0, 0.05, 0.1, 0.08]
        # plt.plot(line_x, line_y, color="crimson")
        #
        # line2_x = [0, 0.05, 0.1, 0.08]
        # line2_y = [0, 0.1, 0.2, 0.3]
        # plt.plot(line2_x, line2_y, color="royalblue")

        # plot target lines
        target_line_dist = 0.7

        line_a_x = [0, 0]
        line_a_y = [0, target_line_dist]

        line_e_x = [0, 0]
        line_e_y = [0, -target_line_dist]

        line_c_x = [0, target_line_dist]
        line_c_y = [0, 0]

        line_g_x = [0, -target_line_dist]
        line_g_y = [0, 0]

        lines_x = [line_a_x, line_c_x, line_e_x, line_g_x]
        lines_y = [line_a_y, line_c_y, line_e_y, line_g_y]

        for x, y in zip(lines_x, lines_y):
            plt.plot(x, y, color='lightsteelblue', linestyle="--")

        # print translations of the trial
        if self.averages:
            avg_translations = self.averages
        else:
            avg_translations = [cw_a, ccw_a]

        for a in avg_translations:
            if a.direction == "cw":
                line_color = "crimson"
            elif a.direction == "ccw":
                line_color = "royalblue"
            else:
                line_color = "black"
            #pdb.set_trace()
            plt.plot([0, a.max_trans_coords[0]], [0, a.max_trans_coords[1]], color=line_color)

if __name__ == '__main__':
    h = AstHandRotation(["sub1", "sub2", "sub3"], "2v2")
    h.filter_data()
    h.plot_ast_avg(subjects=None, show_plot=True, save_plot=False)