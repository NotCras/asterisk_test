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
from alive_progress import alive_bar
import data_manager as datamanager
import matplotlib.pyplot as plt
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
    hand_names = ["2v2", "2v3", "3v3", "barrett",  "m2active", "m2stiff", "basic", "modelvf"]
    # ["basic", "m2active", "2v2", "3v3", "2v3", "barrett", "modelvf"] # "m2stiff",
    rotation_conditions = ["n", "m15", "p15"]
    run_aruco = False
    run_metric_analysis = True
    run_translations = True  # TODO: need to edit num of entries calculation to consider this
    run_standing_rotations = True

    # failed_files = []  # TODO: make a log of everything that happens when data is run

    # [item for item in x if item not in y]
    # z = len(list(set(x) - set(y)))

    # calculations on how many calculation sets to run for alive bar
    len_hands_doing_rotations = len(list(set(hand_names) - set(datamanager.generate_options("hands_only_n"))))
    num_calculation_sets = len(datamanager.generate_options("rotations_n_trans")) * len_hands_doing_rotations + \
                           2 * len(hand_names) + \
                           2 * len(["m15", "p15"]) * len_hands_doing_rotations

    # cw/ccw for all hands that can do rotations +
    # all _n trials for all hands, 1 for generating the AstHandTranslation object and 1 for saving plots
    # all _m15 and _p15 trials for hands that can do rotations
    # if we are doing aruco analysis, then multiply everything by 2 because we have to aruco analyze all of those trials

    if run_aruco:
        entries = 2 * num_calculation_sets
    else:
        entries = num_calculation_sets

    # the actual calculations
    with alive_bar(entries) as bar:
        for h in hand_names:
            print(f"Running: {h}, {subjects}")
            # input("Please press <ENTER> to continue")  # added this for debugging by hand

            if run_aruco:
                print(f"Analyzing aruco codes on {h} viz data...")
                for s in subjects:
                    batch_aruco_analysis(s, h, no_rotations=False, home=home_directory, indices=False, crop=False)
                bar()

            if run_translations:
                for rot in rotation_conditions:
                    if rot in ["m15", "p15"] and h in datamanager.generate_options("hands_only_n"):
                        continue  # skip hands that don't do rotations

                    print(f"Getting {h} ({rot}) data...")
                    data = AstHandTranslation(subjects, h, rotation=rot, blocklist_file="trial_blocklist.csv")
                    # data = study.return_hand(h)
                    bar()

                    print(f"Filtering data...")
                    data.filter_data(10)  # don't use if you're using an asterisk_study obj

                    print("Generating CSVs of paths...")
                    data.save_all_data()

                    print("Calculating averages...")
                    #data.calc_averages(exclude_path_labels=['major deviation'])

                    print("Saving plots...")
                    data.plot_ast_avg(show_plot=False, save_plot=True, exclude_path_labels=['major deviation'])
                    for a in data.averages:
                        a.avg_debug_plot(show_plot=False, save_plot=True, use_filtered=True)
                        #a.save_data(file_name_overwrite=)

                    # although we don't show the plots, a matplotlib warning suggests that it still keeps those plots open
                    plt.close("all")

                    if run_metric_analysis:
                        print("Consolidating metrics together...")
                        results = AstHandAnalyzer(data)

                        print("Saving metric data...")
                        results.save_data(file_name_overwrite=f"{h}_{rot}")

                    print(f"{h} data generation is complete!")
                    # TODO: mean of empty slice error throws here? Probably bar()
                    bar()
                    print("   ")

                    if run_standing_rotations:
                        print(f"Considering cw/ccw for {h}...")
                        for rot2 in datamanager.generate_options("rotations_n_trans"):
                            if rot in ["m15", "p15"]:
                                print(f"Not for {rot} rotation.")
                                continue

                            if h in datamanager.generate_options("hands_only_n"):
                                print("... Nope!")
                                continue

                            print(f"Getting {h} ({rot2}) data...")
                            data_r = AstHandRotation(subjects, h)

                            print(f"Filtering data...")
                            data_r.filter_data(10)

                            print("Generating CSVs of paths...")
                            data_r.save_all_data()

                            print("Calculating averages...")
                            data_r.calc_averages(exclude_path_labels=['major deviation'])

                            print("Saving plots...")
                            data_r.plot_ast_avg(show_plot=False, save_plot=True, exclude_path_labels=['major deviation'])
                            data_r.save_all_data_plots()
                            plt.close("all")

                            if run_metric_analysis:
                                print("Consolidating metrics together...")
                                results = AstHandAnalyzer(data_r, do_avg_line_metrics=False)
                                # TODO: can we do AvgRotation with avg line metrics?

                                print("Saving metric data...")
                                results.save_data(file_name_overwrite=f"{h}_{rot}")

                            bar()


if __name__ == '__main__':
    home_directory = Path(__file__).parent.absolute()

    run_ast_study()
