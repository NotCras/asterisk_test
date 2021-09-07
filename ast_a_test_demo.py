"""
This file serves as an example of how to analyze (and organize) asterisk test data.
The entire pipeline from aruco analysis of vision data to metric calculations.
"""

from pathlib import Path
from ast_hand_translation import AstHandTranslation
from ast_hand_rotation import AstHandRotation
from ast_study import AstStudyTrials
from ast_aruco import batch_aruco_analysis
from metric_analyzers import AstHandAnalyzer
import data_manager as datamanager
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
    subjects = datamanager.generate_options("subjects")  # TODO: debug different hands
    hand_names = ["2v2", "2v3", "3v3", "barrett", "m2active", "basic", "modelvf", "m2stiff"]
    # ["basic", "m2active", "2v2", "3v3", "2v3", "barrett", "modelvf"] # "m2stiff",

    # failed_files = []  # TODO: make a log of everything that happens when data is run

    for h in hand_names:
        print(f"Running: {h}, {subjects}")
        # input("Please press <ENTER> to continue")  # added this for debugging by hand

        # print("Analyzing aruco codes on viz data...")
        # for s in subjects:
        #     batch_aruco_analysis(s, h, no_rotations=False, home=home_directory, indices=False, crop=False)

        for rot in ['n']: #['m15', 'p15']:  #['n']:  # , "m15", "p15"]:
            print(f"Getting {h} ({rot}) data...")
            data = AstHandTranslation(subjects, h, rotation=rot, blocklist_file="trial_blocklist.csv")
            # data = study.return_hand(h)

            print(f"Filtering data...")
            data.filter_data(10)  # don't use if you're using an asterisk_study obj

            print("Generating CSVs of paths...")
            data.save_all_data()

            print("Calculating averages...")
            data.calc_averages()

            print("Saving plots...")

            data.plot_ast_avg(show_plot=False, save_plot=True)
            for a in data.averages:
                a.avg_debug_plot(show_plot=False, save_plot=True, use_filtered=True)

            # print("Consolidating metrics together...")
            # results = AstHandAnalyzer(data)
            #
            # print("Saving metric data...")
            # results.save_data(file_name_overwrite=f"{h}_{rot}")
            #
            # print(f"{h} data generation is complete!")

        for rot2 in datamanager.generate_options("rotations_n_trans"):
            print(f"Getting {h} ({rot2}) data...")
            data = AstHandRotation(subjects, h)

            print(f"Filtering data...")
            data.filter_data(10)

            print("Generating CSVs of paths...")
            data.save_all_data()

            print("Calculating averages...")
            data.calc_averages()

            print("Saving plots...")
            data.plot_ast_avg(show_plot=False, save_plot=True)
            data.save_all_data_plots()

    # print("Getting subplot figures, using Asterisk Study obj")
    # # I know this is stupidly redundant, but for my purposes I can wait
    # study = AsteriskStudy(subjects_to_collect=subjects, hands_to_collect=hand_names, rotation="n")
    # study.filter_data(window_size=25)
    # study.plot_all_hands(rotation="n", show_plot=True, save_plot=True)


if __name__ == '__main__':
    home_directory = Path(__file__).parent.absolute()

    run_ast_study()
