import pdb
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

from file_manager import AstDirectory
from ast_hand_translation import AstHandTranslation
from data_plotting import AsteriskPlotting as aplt

home_directory = Path("/home/john/Programs/new_ast_data")
data_directory = home_directory
new_ast_files = AstDirectory()
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
for rotation_type in ["x", "p15", "m15"]:
    for hand in ["2v2", "2v3", "3v3", "p2vp2", "2v1", "p1vp1"]:
        print(hand)
        hand_data = AstHandTranslation(new_ast_files, hand_name=hand, rotation=rotation_type)
        hand_data.load_trials()
        print(list(hand_data.data.keys()))
        hand_data.filter_data()

        trials = hand_data.data
        aplt.plot_asterisk(new_ast_files, dict_of_trials=trials, hand_name=hand, show_plot=False, save_plot=True)

        hand_data.calc_averages(save_debug_plot=True)
        hand_data.plot_avg_asterisk(show_plot=False, save_plot=True)
        hand_data.save_all_data()
        print("======")
        print(" ")

        plt.close("all")

