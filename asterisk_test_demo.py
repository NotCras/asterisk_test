"""
This file serves as an example of how to analyze (and organize) asterisk test data.
The entire pipeline from aruco analysis of vision data to metric calculations.
"""

from pathlib import Path
from ast_hand_data import AstHandTrials
from ast_study import AstStudyTrials
from ast_aruco import batch_aruco_analysis
from metric_analyzers import AstHandAnalyzer
import asterisk_data_manager as datamanager
import ast_trial as t


def run_ast_study():
    """
    Handles running an asterisk study
    1) imports all specified data, runs aruco analysis
    2) filters everything with a 15 sample moving average
    3) averages all similar trials
    4) produces a final averaged plot for each hand, as well as by subject
    5) writes salient data tp external files (averaged asterisk data, final statistics, generated plots)
    """

    home_directory = Path(__file__).parent.absolute()

    # right now, just compiles data and saves it all using the AsteriskHandData object
    subjects = datamanager.generate_options("subjects")
    hand_names = ["basic", "m2active", "2v2", "3v3", "2v3", "barrett", "modelvf"]  # "m2stiff",

    # failed_files = []  # TODO: add ability to collect failed files

    for h in hand_names:
        print(f"Running: {h}, {subjects}")
        # input("Please press <ENTER> to continue")  # added this for debugging by hand

        print("Analyzing aruco codes on viz data...")
        for s in subjects:
            batch_aruco_analysis(s, h, no_rotations=False, home=home_directory)

        for rot in ['n']:  # , "m15", "p15"]:
            print(f"Getting {h} ({rot}) data...")
            # data = AstHandTrials(subjects, h, rotation="n", blocklist_file="trial_blocklist.csv")
            data = AstHandTrials(subjects, h, rotation=rot, blocklist_file="trial_blocklist.csv")
            # data = study.return_hand(h)

            print(f"Getting {h} data...")
            data.filter_data(10)  # don't use if you're using an asterisk_study obj

            print("Generating CSVs of paths...")
            data.save_all_data()

            print("Calculating averages...")
            data.calc_averages( rotation="n")

            print("Saving plots...")

            data.plot_ast_avg(rotation=rot, show_plot=False, save_plot=True)
            for a in data.averages:
                a.avg_debug_plot(show_plot=False, save_plot=True, use_filtered=True)

            print("Consolidating metrics together...")
            results = AstHandAnalyzer(data)

            print("Saving metric data...")
            results.save_data(file_name_overwrite=f"{h}_{rot}")

            print(f"{h} data generation is complete!")

    # print("Getting subplot figures, using Asterisk Study obj")
    # # I know this is stupidly redundant, but for my purposes I can wait
    # study = AsteriskStudy(subjects_to_collect=subjects, hands_to_collect=hand_names, rotation="n")
    # study.filter_data(window_size=25)
    # study.plot_all_hands(rotation="n", show_plot=True, save_plot=True)


if __name__ == '__main__':
    home_directory = Path(__file__).parent.absolute()

    run_ast_study()
