#!/usr/bin/env python3
"""
Class for organizing asterisk trial data for one specific hand. Handles analysis, averaging, and plotting.
"""
import logging
import os

import numpy as np
import pandas as pd
import math as m
from pathlib import Path
import csv
import pdb
import matplotlib.pyplot as plt
import data_manager as datamanager
from ast_trial_translation import AstTrialTranslation
from ast_hand_info import HandInfo
from ast_avg_translation import AveragedTranslationTrial
from data_plotting import AsteriskPlotting as aplt
from file_manager import AstDirectory, my_ast_files


class AstHandTranslation:
    def __init__(self, file_loc_obj, hand_name, rotation='n', blocklist_file=None, normalized_data=True):
        """
        Class to hold all the data pertaining to a specific hand.
        Combines data from all subjects
        :param subjects: subjects to include in hand data object
        :param hand_name: name of hand for this object
        """
        self.hand = HandInfo(hand_name)
        self.subjects_containing = set()
        self.aruco_id = None
        self.file_locs = file_loc_obj

        if blocklist_file is not None:
            self.blocklist = self._check_blocklist(blocklist_file)
        else:
            self.blocklist = None

        self.rotation_type = rotation
        self.data = {}
        self.filtered = False
        self.window_size = None
        self.averages = {}

    def _check_blocklist(self, file_name):
        """
        Checks blocklist file to get the list of trials that should not be included.
        """
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            blocked_files = []
            for row in csv_reader:
                blocked_files.append(row[0])

            logging.info(f"Will block these files: {blocked_files}")

        return blocked_files

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

            if r != self.rotation_type:
                continue

            if subjects_to_exclude is not None:
                if s in subjects_to_exclude:
                    continue

            if trial_num_to_exclude is not None:
                if n in trial_num_to_exclude:
                    continue

            if directions_to_exclude is not None:
                if t in directions_to_exclude or r in directions_to_exclude:  # TODO: just trying to capture cw/ccw here
                    continue

            if self.blocklist is not None:
                if file_name in self.blocklist:  # TODO: make sure that the blocklist is implemented correctly
                    continue

            logging.info(f"{h}_{t}_{r}_{s}_{n} has passed!")

            label = f"{t}_{r}"
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
            label = f"{t}_{r}"  # TODO: actually... don't need r here, right?

            self.data[label].append(trial)  # TODO: its up to the user to make sure data doubles don't happen?
            self.subjects_containing.add(s)

    def load_aruco_file(self, file_name):
        """
        Makes an AstTrialTranslation from an aruco_loc file
        """
        trial = AstTrialTranslation(file_loc_obj=self.file_locs)
        trial.load_data_by_aruco_file(file_name)

        return trial

    def get_data_from_files(self, subjects, blocklist=None, normalized_data=True):
        """
        Returns a dictionary with the data for the hand, sorted by task.
        Each key,value pair of dictionary is:
        key: name of task, string. Ex: "a_n"
        value: list of AsteriskTrial objects for the corresponding task, with all subjects specified
        :param subjects: list of subjects to get
        """ # TODO: change this to add trials to the data dictionary. If the key doesn't exist, then add it. Add it all directly to the self.data attribute
        data_dictionary = dict()

        for t in datamanager.get_option_list("translations"):
            key = f"{t}_{self.rotation_type}"
            data = self._make_asterisk_trials_from_filenames(subjects, t, self.rotation_type,
                                                             datamanager.get_option_list("numbers"),
                                                             blocklist=self.blocklist, norm_data=normalized_data)
            if data:
                data_dictionary[key] = data
                # pdb.set_trace()
            else:
                logging.error(f"{key} not included, no valid data")
                # pdb.set_trace()

        return data_dictionary  # TODO: remove this!

    def _get_directions_in_data(self):
        """
        Returns a list of trial directions that exist in the data
        :return:
        """
        list_of_dirs = list()
        for k in list(self.data.keys()):
            t, r = k.split("_")
            if t != "n":
                list_of_dirs.append(t)

        return list_of_dirs

    def _make_asterisk_trials_from_filenames(self, subjects, translation_label, rotation_label, trial_nums,
                                             blocklist=None, norm_data=True):
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
            for n in trial_nums:
                asterisk_trial = f"{s}_{self.hand.get_name()}_{translation_label}_{rotation_label}_{n}"

                if blocklist is not None and asterisk_trial in blocklist:
                    logging.warning(f"{asterisk_trial} is blocklisted and will not be used.")
                    continue

                try:
                    trial_data = AstTrialTranslation(self.file_locs)
                    trial_data.load_data_by_aruco_file(f"{asterisk_trial}.csv",
                                                       norm_data=norm_data, condition_data=True, do_metrics=True)
                    logging.info(f"{trial_data.generate_name()}, labels: {trial_data.path_labels}")

                    gathered_data.append(trial_data)

                except Exception as e:
                    logging.error(f"AstTrial generation failed for {asterisk_trial}")
                    logging.error(e)
                    continue

        return gathered_data

    def get_data_from_arucolocs(self, arucoloc_list):
        """

        Args:
            arucoloc_list (_type_): _description_
        """
        pass  # TODO: flesh this out after I tackle filenames

    def _make_asterisk_trials_from_arucoloc(self, arucoloc_list, blocklist=None, norm_data=True):
        """_summary_

        Args:
            arucoloc_list (_type_): _description_
            blocklist (_type_, optional): _description_. Defaults to None.
            norm_data (bool, optional): _description_. Defaults to True.
        """

        # for each arucoloc obj, we need to get its attributes and find the appropriate location to store the data
        # then we need to make the asterisk trial obj from the arucoloc obj
        for al in arucoloc_list:
            t_label = al.data_attributes["translation"]
            r_label = al.data_attributes["rotation"]
            trial_label = f"{t_label}_{r_label}"

            trial = AstTrialTranslation(self.file_locs)
            trial.add_data_by_arucoloc(al)

            self.data[trial_label].append(trial)


    def _import_data_from_ast_trial_list(self, trial_list):  # TODO: implement!
        """Adds data from a list of asterisk trial objects

        Args:
            trial_list (_type_): _description_
        """
        pass

    def add_trial(self, ast_trial):
        """
        add an ast_trial after the asteriskhanddata object was created
        :param ast_trial: asterisktrialdata to add
        """
        # TODO: check that the hand is correct!
        if ast_trial.hand.get_name() == self.hand.get_name() and ast_trial.trial_rotation == self.rotation_type:
            label = f"{ast_trial.trial_translation}_{ast_trial.trial_rotation}"
            self.data[label].append(ast_trial)
        else:
            logging.warning(f"Did not load AstTrial {ast_trial.generate_name()} because it did not fit in this hand obj.")

    def _get_ast_dir(self, direction_label, subjects, exclude_path_labels=None):
        """
        Get all of the trials for a specific direction. You can specify subject too
        :param direction_label: translation direction
        :param subjects: subject or list of subjects to include
        :param rotation_label: rotation label, defaults to "n"
        """
        dict_key = f"{direction_label}_{self.rotation_type}"
        direction_trials = self.data[dict_key]

        gotten_trials = []
        dont_include = False

        #pdb.set_trace()

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

    def replace_trial_data(self, trial_obj):
        """
        Delete trial data obj from stored data and replace with new trial data obj
        Gets attributes of obj to delete from the obj passed in
        """
        # TODO: implement this
        pass

    def _parse_trial_name(self, trial_name):
        """Takes a filename and returns the dict key that it is in

        Args:
            trial_name (string): trial name string
        """
        _, t, r, s, n = trial_name.split("_")
        return f"{t}_{r}", s, n

    def _get_trial(self, trial_name):
        """Gets a trial out of data object. If trial doesn't exist, will throw an error

        Args:
            trial_name (string): trial name string
        """
        key, s, n = self._parse_trial_name(trial_name)

        trials = self.data[key]

        trial_to_return = None
        for t in trials:
            if t.subject == s and t.trial_num == n:
                trial_to_return = t
                break

        if trial_to_return is None:
            raise AttributeError(f"Could not find the trial you specified: {trial_name}")
        else:
            return trial_to_return

    def _average_dir(self, translation, subject=None, exclude_path_labels=None):
        """
        Averages a set of asterisk_trial paths. We run this on groups of paths of the same direction.
        :param translation: trial direction to average
        :param rotation: trial rotation to average
        :param subject: subject or list of subjects to average, optional. If not provided, defaults to all subjects
        :return returns averaged path
        """
        if subject is None:  # get batches of data by trial type, if no subjects given, defaults to all subjects
            trials = self._get_ast_dir(direction_label=translation, subjects=self.subjects_containing,
                                       exclude_path_labels=exclude_path_labels)

        else:
            trials = self._get_ast_dir(direction_label=translation, subjects=subject,
                                       exclude_path_labels=exclude_path_labels)

        if trials:
            average = AveragedTranslationTrial(file_obj=self.file_locs, trials=trials)
            return average

        else:
            print(f"No trials for {translation}_{self.rotation_type}, skipping averaging.")
            return None

    def calc_averages(self, subjects=None, exclude_path_labels=None, save_debug_plot=False, show_debug_plot=False):
        """
        calculate and store all averages
        :param subjects: subject(s) to include in the average. Defaults to all subjects in object
        :param rotation: refers to the rotation type ("n", "m15", "p15"). Defaults to all options
        """
        #averages = []  # TODO: make this a dictionary for easier access?
        averages = {}
        if subjects is None:  # if no subjects given, defaults to all subjects
            subjects = self.subjects_containing

        dirs = self._get_directions_in_data()
        logging.info(f"Calculating averages for: {dirs}")

        for t in datamanager.get_option_list("translation_only"):
            # make sure that we only include translations that are in the data
            if t in dirs:  # TODO: why don't I just iterate through dirs?
                logging.info(f"Averaging {t}")
                avg = self._average_dir(translation=t, subject=subjects,
                                        exclude_path_labels=exclude_path_labels)
                if avg is not None:
                    label = f"{t}_{self.rotation_type}"
                    averages[label] = [avg]  # it is put in a list so it works with aplt.plot_asterisk()

                    if save_debug_plot or show_debug_plot:
                        avg.avg_debug_plot(show_plot=show_debug_plot, save_plot=save_debug_plot)

        self.averages = averages
        return averages

    def get_averages(self, directions_to_include=None, path_labels_to_exclude=None):
        """
        Calculate and store all averages.
        """
        avg_dict = {}

        if directions_to_include is not None:
            dirs = self._get_directions_in_data()
        else:
            dirs = directions_to_include

        for t in dirs:
            label = f"{t}_{self.rotation_type}"

            avg = self._average_dir(translation=t, exclude_path_labels=path_labels_to_exclude)

            if avg is not None:
                avg_dict[label] = [avg]

        self.averages = avg_dict

        return avg_dict

    def filter_data(self, window_size=15):
        """
        Runs moving average on data stored inside object
        :param window_size: size of moving average. default is 15
        """
        for key in self.data.keys():
            for t in self.data[key]:
                print(f"Moving Average of size {window_size} is on {t.generate_name()}")
                t.moving_average(window_size)

        self.filtered = True
        self.window_size = window_size

    def save_all_data(self):
        """
        Saves each AsteriskTrialObject as a csv file
        """
        for key in self.data.keys():
            for t in self.data[key]:
                t.save_data()
                # print(f"Saved: {t.generate_name()}")

    def save_all_data_plots(self, use_filtered=True, provide_notes=True):
        """
        Saves each AsteriskTrialObject plot as a jpg file
        """
        for key in self.data.keys():
            for t in self.data[key]:
                t.plot_trial(use_filtered=use_filtered, provide_notes=provide_notes, show_plot=False, save_plot=True)
                # print(f"Saved: {t.generate_name()}")

    def _make_plot(self, trials, use_filtered=True, stds=False, linestyle="solid", picky_tlines=False,
                   td_labels=True, incl_obj_img=True, include_notes=True, plot_contributions=False):
        # TODO: add new parameters to plot_avg_ast function
        """
        Function to make our plots.
        :param trials: either a list of AsteriskTrialData or AsteriskAverage objs
        :param use_filtered: flag whether to use filtered data. Default is True
        :param stds: flag whether to plot standard deviations. Only for AsteriskAverage objects. Default is False
        """
        # TODO: plot orientation error?

        #plt.figure(figsize=(7, 7))
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)

        # get all the averages that we have
        dir_labels = set()
        for a in trials:  # TODO: what if we have no averages?
            dir_labels.add(a.trial_translation)

        if picky_tlines and len(dir_labels) < 8:
            # plot target lines as dotted lines
            aplt.plot_all_target_lines(specific_lines=list(dir_labels))
        else:
            aplt.plot_all_target_lines()

        # plot data
        for i, t in enumerate(trials):
            data_x, data_y, theta = t.get_poses(use_filtered)

            # ax.plot(data_x, data_y, color=colors[i], label='trajectory', linestyle=linestyle)
            ax.plot(data_x, data_y, color=aplt.get_dir_color(t.trial_translation),
                    label='trajectory', linestyle=linestyle)

            # plot orientation error
            t._plot_orientations(marker_scale=15, line_length=0.025, scale=1)

            # plot total_distance value in each direction
            if td_labels:
                aplt.add_dist_label(t, ax=ax)

            if stds and t.is_avg_trial():  # only for AsteriskAverage objs
                t.plot_sd(aplt.get_dir_color(t.trial_translation))

        if include_notes:
            aplt.plot_notes(trials, ax=ax)

        if incl_obj_img:
            aplt.add_obj_img(self.rotation_type, fig)

        if plot_contributions:
            for a in trials:
                a._plot_line_contributions()

        fig.suptitle(f"{self.hand.get_name()}, {self.rotation_type} Avg Asterisk", fontweight="bold", fontsize=14)
        ax.set_title("Cube size: ~0.25 span, init pos: 0.75 depth")  #, pad=10)
        #plt.title(f"{self.hand.get_name()} avg asterisk")  # , rot: {trials[0].trial_rotation}")
        ax.axis([-0.7, 0.7, -0.7, 0.7])
        ax.tick_params(axis="x", rotation=30)
        # plt.xticks(np.linspace(-0.7, 0.7, 15), rotation=30)
        # plt.yticks(np.linspace(-0.7, 0.7, 15))
        plt.gca().set_aspect('equal', adjustable='box')
        return plt

    def plot_specific_trials(self, trial_list, show_plot=True, save_plot=False, include_notes=True,
                     linestyle="solid", plot_contributions=False, exclude_path_labels=None,
                     picky_tlines=False, td_labels=True, incl_obj_img=True):
        """
        Give it a list of trials to plot and this will single out those trials and plot it on a full asterisk plot
        """
        selected_trial_objs = []

        for t in trial_list: # TODO: skip the trials that do not pertain to this AstHandTranslation trial
            selected_trial_objs.append(self._get_trial(t))

        plt = self._make_plot(selected_trial_objs, use_filtered=False, stds=False, linestyle=linestyle, include_notes=include_notes,
                              picky_tlines=picky_tlines, td_labels=td_labels,
                              incl_obj_img=incl_obj_img, plot_contributions=plot_contributions)

        # TODO: add orientation markers to each line so we have some idea of orientation along the path
        # TODO: add attributes for object shape, size, and initial position!
        #plt.title(f"Avg {self.hand.get_name()}, {subjects}, {self.set_rotation} \n Cube (0.25 span), 0.75 depth init pos")

        if save_plot:
            plt.savefig(self.file_locs.result_figs / f"selected_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.rotation_type}.jpg", format='jpg')
            #plt.savefig(f"results/pics/selected/selected_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.set_rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            logging.info("Figure saved.")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def plot_ast_avg(self, subjects=None, show_plot=True, save_plot=False, include_notes=True,
                     linestyle="solid", plot_contributions=False, exclude_path_labels=None,
                     picky_tlines=False, td_labels=True, incl_obj_img=True):
        """
        Plots the data from one subject, averaging all of the data in each direction
        :param subjects: list of subjects. If none is provided, uses all of them
        :param rotation: the type of rotation type to plot, will collect an asterisk of this
        :param show_plot: flag to show plot. Default is true
        :param save_plot: flat to save plot as a file. Default is False
        """

        # TODO: should we do the same for filtered vs unfiltered?
        # if self.averages and subjects is None:
        #     # if we have averages and the user does not specify subjects just use the averages we have
        #     subjects = self.subjects_containing
        #     avgs = self.averages

        if self.averages and subjects is not None:
            # if we have averages but the user specifies specific subjects, rerun averaage
            avgs = self.calc_averages(subjects=subjects, exclude_path_labels=exclude_path_labels)

        else:
            # otherwise just run the average on everything
            subjects = self.subjects_containing
            avgs = self.calc_averages(subjects=subjects, exclude_path_labels=exclude_path_labels)

        plt = self._make_plot(avgs, use_filtered=False, stds=True, linestyle=linestyle, include_notes=include_notes,
                              picky_tlines=picky_tlines, td_labels=td_labels,
                              incl_obj_img=incl_obj_img, plot_contributions=plot_contributions)

        # TODO: add orientation markers to each line so we have some idea of orientation along the path
        # TODO: add attributes for object shape, size, and initial position!
        #plt.title(f"Avg {self.hand.get_name()}, {subjects}, {self.set_rotation} \n Cube (0.25 span), 0.75 depth init pos")

        if save_plot:
            plt.savefig(self.file_locs.result_figs / f"avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.rotation_type}.jpg", format='jpg')
            #plt.savefig(f"results/pics/avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.set_rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            logging.info("Figure saved.")

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

    def plot_avg_asterisk(self, show_plot=True, save_plot=False, include_notes=True, show_avg_deviation=True,
                          linestyle="solid", plot_contributions=False, exclude_path_labels=None, plot_orientations=False,
                          picky_tlines=False, tdist_labels=True, incl_obj_img=True):

        # generate averages (or get existing ones)
        averages = self.averages

        # throw averages into plot_asterisk  # TODO: make an option to plot the avg paths ontop of the trials, greyed out?
        ax, fig = aplt.plot_asterisk(self.file_locs, averages, #plotting_averages_with_sd=True,
                                rotation_condition=self.rotation_type, hand_name=self.hand.get_name(),
                                use_filtered=True, tdist_labels=tdist_labels, include_notes=include_notes,
                                linestyle=linestyle, plot_orientations=plot_orientations, incl_obj_img=incl_obj_img,
                                save_plot=False, show_plot=False)

        """
        (file_loc, dict_of_trials, rotation_condition="x", hand_name="",
        use_filtered=True, linestyle="solid",
        include_notes=False, labels=None,
        plot_orientations=False, tdist_labels=True,
        incl_obj_img=True, gray_it_out=False,
        save_plot=False, show_plot=True):
        """

        # # TODO: add in the average deviation regions
        if show_avg_deviation:
            for a_k in list(averages.keys()):
                a_trial = averages[a_k][0]

                aplt.plot_sd(ax, a_trial, aplt.get_dir_color(a_trial.trial_translation))

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

        if save_plot:
            plt.savefig(self.file_locs.result_figs / f"avg_ast_{self.hand.get_name()}_{self.rotation_type}.jpg", format='jpg')
            #plt.savefig(self.file_locs.result_figs / f"avg_ast_{self.hand.get_name()}_{self.set_rotation}.jpg", format='jpg')
            #plt.savefig(f"results/pics/avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.set_rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            logging.info("Figure saved.")


if __name__ == '__main__':
    home_directory = Path("/home/john/Programs/new_ast_data")
    data_directory = home_directory / "data"
    new_ast_files = AstDirectory()
    new_ast_files.data_home = data_directory
    new_ast_files.compressed_data = data_directory / "compressed_data"
    new_ast_files.aruco_pics = data_directory / "viz"
    new_ast_files.aruco_data = data_directory / "aruco_data"
    new_ast_files.path_data = data_directory / "trial_paths"
    new_ast_files.metric_results = data_directory / "results"
    new_ast_files.result_figs = data_directory / "results" / "plots"
    new_ast_files.debug_figs = data_directory / "results" / "debug_plots"
    new_ast_files.resources = data_directory.parent / "resources"

    logging.basicConfig(level=logging.WARNING)

    hand_data = AstHandTranslation(my_ast_files, hand_name="2v2", rotation="n")
    hand_data.load_trials()
    print(list(hand_data.data.keys()))
    hand_data.filter_data()

    # # subject 1 averages
    # h.plot_avg_data(rotation="n", subjects="sub1", show_plot=False, save_plot=True)
    # plt.clf()
    #
    # # subject 2 averages
    # h.plot_avg_data(rotation="n", subjects="sub2", show_plot=False, save_plot=True)
    # plt.clf()
    #
    # # all subjects
    #h.plot_ast_avg(subjects=None, show_plot=True, save_plot=False)

    trials = hand_data.data
    aplt.plot_asterisk(my_ast_files, dict_of_trials=trials)

    hand_data.calc_averages(exclude_path_labels=["end deviated", ])
    hand_data.plot_avg_asterisk()


