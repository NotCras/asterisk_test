import pdb
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

from file_manager import AstDirectory
from ast_hand_translation import AstHandTranslation
from ast_hand_rotation import AstHandRotation
from metric_analyzers import AstHandAnalyzer
from data_plotting import AsteriskPlotting as aplt

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

#rotation_type = "x"
for hand in ["2v2", "2v3", "3v3", "p2vp2"]:
    for rotation_type in ["x", "p15", "m15"]:
        print(hand)
        hand_data = AstHandTranslation(new_ast_files, hand_name=hand, rotation=rotation_type)
        hand_data.load_trials()
        print(list(hand_data.data.keys()))
        hand_data.filter_data()

        trials = hand_data.data
        aplt.plot_asterisk(new_ast_files, dict_of_trials=trials, hand_name=hand, show_plot=False, save_plot=True)

        hand_data.calc_averages(exclude_path_labels=["end deviated", "deviated", "rot deviated"], save_debug_plot=True)
        hand_data.plot_avg_asterisk(show_plot=False, save_plot=True)
        hand_data.save_all_data()
        print("======")
        print(" ")

        plt.close("all")

        metric_results = AstHandAnalyzer(new_ast_files, hand_data)
        metric_results.save_data()

    # also do rotation only trials
    hand_rot_data = AstHandRotation(new_ast_files, hand_name=hand)
    hand_rot_data.load_trials()
    hand_rot_data.filter_data()
    hand_rot_data.calc_averages(exclude_path_labels=["too deviated"])
    hand_rot_data.plot_avg_asterisk(show_plot=False, save_plot=True)

    metric_results = AstHandAnalyzer(new_ast_files, hand_rot_data, do_avg_line_metrics=False)
    metric_results.save_data()


for hand in ["2v1", "p1vp1"]:
    print(hand)
    hand_data = AstHandTranslation(new_ast_files, hand_name=hand, rotation="x")
    hand_data.load_trials()
    print(list(hand_data.data.keys()))
    hand_data.filter_data()

    trials = hand_data.data
    aplt.plot_asterisk(new_ast_files, dict_of_trials=trials, hand_name=hand, show_plot=False, save_plot=True)

    hand_data.calc_averages(exclude_path_labels=["end deviated", "deviated", "rot deviated"], save_debug_plot=True)
    hand_data.plot_avg_asterisk(show_plot=False, save_plot=True)
    hand_data.save_all_data()
    print("======")
    print(" ")

    plt.close("all")

    metric_results = AstHandAnalyzer(new_ast_files, hand_data)
    metric_results.save_data()
