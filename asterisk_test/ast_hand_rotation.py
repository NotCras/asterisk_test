import pdb
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from pathlib import Path

from ast_hand_translation import AstHandTranslation
from ast_avg_rotation import AveragedRotationTrial
import data_manager as datamanager
from ast_trial_rotation import AstTrialRotation
from file_manager import my_ast_files, AstDirectory
from data_plotting import AsteriskPlotting as aplt


class AstHandRotation(AstHandTranslation):
    def __init__(self, file_obj, hand_name):
        super().__init__(file_loc_obj=file_obj, hand_name=hand_name, rotation=None)
        # TODO: is it ok to just leave set_rotation var hanging around?

    def load_trials(self, data_type="aruco_data", directions_to_exclude=None, subjects_to_exclude=None, trial_num_to_exclude=None):
        """
        Searches the file location indicated by data_loc (either "aruco_data" or "trial_paths")
        """

        if data_type == "aruco_data":
            folder_to_check = self.file_locs.aruco_data
        elif data_type == "trial_paths":
            folder_to_check = self.file_locs.path_data
        else:
            raise FileNotFoundError("Incorrect data type specified.")

        data_files = [f for f in os.listdir(folder_to_check) if f[-3:] == 'csv']

        for file_name in data_files:
            # parse filename
            h, t, r, s, e = file_name.split("_")
            n, _ = e.split(".")
            logging.info(f"Considering: {h}_{t}_{r}_{s}_{n}")

            if h != self.hand.get_name():  # TODO: change to target_hand
                continue  # skip this hand

            if t != "x":
                continue

            if subjects_to_exclude is not None:
                if s in subjects_to_exclude:
                    continue

            if trial_num_to_exclude is not None:
                if n in trial_num_to_exclude:
                    continue

            if self.blocklist is not None:
                if file_name in self.blocklist:  # TODO: make sure that the blocklist is implemented correctly
                    continue

            logging.info(f"{h}_{t}_{r}_{s}_{n} has passed!")

            label = f"{r}"
            if label not in list(self.data.keys()):
                self.data[label] = []

            # Now we are sure that this trial is one that we want
            if data_type == "aruco_data":
                trial = self.load_aruco_file(file_name)
            # elif data_type == "trial_paths":
            #     trial.add_data_by_df() # TODO: not actually correct, need a new function for this
            else:
                pass
                raise TypeError("Incorrect aruco_data.")

            # now, add this to the data dict
            label = f"{r}"

            self.data[label].append(trial)  # TODO: its up to the user to make sure data doubles don't happen?
            self.subjects_containing.add(s)

    def load_aruco_file(self, file_name):
        """
        Makes an AstTrialTranslation from an aruco_loc file
        """
        trial = AstTrialRotation(file_loc_obj=self.file_locs)
        trial.load_data_by_aruco_file(file_name)

        return trial

    def _get_directions_in_data(self):
        """
        Returns a list of trial directions that exist in the data
        :return:
        """
        list_of_dirs = list()
        for k in list(self.data.keys()):
            r = k.split("_")

            list_of_dirs.append(r)

        return list_of_dirs

    def _get_ast_dir(self, direction_label, subjects, exclude_path_labels=None):
        """
        Get all of the trials for a specific direction. You can specify subject too
        :param direction_label: translation direction
        :param subjects: subject or list of subjects to include
        :param rotation_label: rotation label, defaults to "n"
        """
        dict_key = f"{direction_label}"
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
            average = AveragedRotationTrial(direction=direction, trials=trials)
            return average

        else:
            print(f"No trials for n_{direction}, skipping averaging.")
            return None

    def calc_averages(self, subjects=None, exclude_path_labels=None, save_debug_plot=False, show_debug_plot=False):
        """
        calculate and store all averages
        :param subjects: subject(s) to include in the average. Defaults to all subjects in object
        :param rotation: refers to the rotation type ("n", "m15", "p15"). Defaults to all options
        """
        averages = {}
        if subjects is None:  # if no subjects given, defaults to all subjects
            subjects = self.subjects_containing

        dirs = self.data.keys()  #self._get_directions_in_data()
        logging.info(f"Calculating averages for: {dirs}")

        for r in datamanager.get_option_list("rotation_only"):
            # make sure that we only include translations that are in the data
            if r in dirs:  # TODO: why don't I just iterate through dirs?
                logging.info(f"Averaging {r}")
                avg = self._average_dir(direction=r, subject=subjects,
                                        exclude_path_labels=exclude_path_labels)
                if avg is not None:
                    label = f"{r}"
                    averages[label] = [avg]  # it is put in a list so it works with aplt.plot_asterisk()

        self.averages = averages
        return averages

    # def plot_ast_avg(self, subjects=None, show_plot=True, save_plot=False, include_notes=True,
    #                  linestyle="solid", plot_contributions=False, exclude_path_labels=None):
    #
    def plot_avg_asterisk(self, show_plot=True, save_plot=False, include_notes=True, show_avg_deviation=True,
                          linestyle="solid", plot_contributions=False, exclude_path_labels=None, plot_orientations=False,
                          picky_tlines=False, tdist_labels=True, incl_obj_img=True):

        fig_size = 7
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw=dict(aspect="equal"))

        subjects = self.subjects_containing

        cw_rot = None
        ccw_rot = None
        cw_rot_std = None
        ccw_rot_std = None
        cw_a = None
        ccw_a = None

        if self.averages:
            for a_key in list(self.averages.keys()):
                a = self.averages[a_key][0]
                if a.direction == "pp":
                    cw_a = a
                    cw_rot = np.abs(a.max_rot[0])
                    cw_rot_std = a.max_rot[1]
                elif a.direction == "mm":
                    ccw_a = a
                    ccw_rot = np.abs(a.max_rot[0])
                    ccw_rot_std = a.max_rot[1]
                else:
                    # TODO: throw an error here?
                    raise TypeError("We have an average stored in this object that doesn't belong!")

            if cw_rot is None or ccw_rot is None:
                print(f"Error with existing averages. cw:{cw_rot}, ccw:{ccw_rot}")

        else:
            cw_a = self._average_dir("pp", subject=subjects)
            cw_rot = np.abs(cw_a.max_rot[0])
            cw_rot_std = cw_a.max_rot[1]

            ccw_a = self._average_dir("mm", subject=subjects)
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
            aplt.plot_notes(cw_a.path_labels, ax=ax)
            aplt.plot_notes(ccw_a.path_labels, ax=ax)

        self._plot_translations()

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        # plt.tight_layout()

        fig.suptitle(f"Rotation Only Averaged: {self.hand.get_name()}", fontweight='bold')
        ax.set_title("Cube size: ~0.25 span, init pos: 0.75 depth")  #, pad=10)

        if save_plot:
            plt.savefig(self.file_locs.result_figs / f"rot_only_avg_{self.hand.get_name()}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def _plot_translations(self):
        # line_x = [0, 0.1, 0.2, 0.3]
        # line_y = [0, 0.05, 0.1, 0.08]
        # plt.plot(line_x, line_y, color="crimson")
        #
        # line2_x = [0, 0.05, 0.1, 0.08]
        # line2_y = [0, 0.1, 0.2, 0.3]
        # plt.plot(line2_x, line2_y, color="royalblue")

        cw_trials = self.data["pp"]
        ccw_trials = self.data["mm"]

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
        # if self.averages:
        #     avg_translations = self.averages
        # else:
        #     avg_translations = [cw_a, ccw_a]
        #
        # for a in avg_translations:
        #     if a.direction == "pp":
        #         line_color = "crimson"
        #     elif a.direction == "mm":
        #         line_color = "royalblue"
        #     else:
        #         line_color = "black"
        #     #pdb.set_trace()
        #     plt.plot([0, a.max_trans_coords[0]], [0, a.max_trans_coords[1]], color=line_color)

        for cw in cw_trials:
            cw._plot_translations(color="crimson")

        for ccw in ccw_trials:
            ccw._plot_translations(color="royalblue")


if __name__ == '__main__':
    home_directory = Path("/home/john/Programs/new_ast_data")
    data_directory = home_directory
    new_ast_files = AstDirectory(home_directory)
    new_ast_files.data_home = data_directory
    new_ast_files.compressed_data = data_directory / "compressed_data"
    new_ast_files.aruco_pics = data_directory / "viz"
    new_ast_files.aruco_data = data_directory / "aruco_data"
    new_ast_files.path_data = data_directory / "trial_paths"
    new_ast_files.metric_results = data_directory / "results"
    new_ast_files.result_figs = data_directory / "results" / "plots"
    new_ast_files.debug_figs = data_directory / "results" / "debug_plots"

    resources_home = Path(__file__).parent.parent.absolute()
    new_ast_files.resources = resources_home.parent / "resources"

    logging.basicConfig(level=logging.WARNING)

    rot_hand = AstHandRotation(new_ast_files, hand_name="2v2")
    rot_hand.load_trials()
    rot_hand.filter_data()
    rot_hand.calc_averages(exclude_path_labels=["too deviated"])
    rot_hand.plot_avg_asterisk(show_plot=True, save_plot=False)

