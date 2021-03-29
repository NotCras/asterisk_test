from pathlib import Path
import asterisk_hand_data as h
from asterisk_hand_data import AsteriskHandData
from asterisk_study import AstHandAnalyzer
from asterisk_aruco import batch_aruco_analysis
import asterisk_data_manager as datamanager
import asterisk_trial as t

def run():
    """
    Handles running an asterisk study
    1) imports all specified data
    2) filters everything with a 15 sample moving average
    3) averages all similar trials
    4) produces a final averaged plot for each hand, as well as by subject
    5) writes salient data tp external files (averaged asterisk data, final statistics, generated plots)
    """

    # right now, just compiles data and saves it all using the AsteriskHandData object
    subjects = datamanager.generate_options("subjects")
    hand_names = ["2v2"]  #["basic", "m2stiff", "m2active", "2v2", "3v3", "2v3", "barrett", "modelvf"]

    # failed_files = []  # TODO: add ability to collect failed files
    for h in hand_names:
        print(f"Running: {h}, {subjects}")
        input("Please press <ENTER> to continue")  # added this for debugging by hand

        print("Analyzing aruco codes on viz data...")
        for s in subjects:
            batch_aruco_analysis(s, h, no_rotations=True)

        print(f"Getting {h} data...")
        data = AsteriskHandData(subjects, h, rotation="n")

        print(f"Getting {h} data...")
        data.filter_data()

        print("Generating CSVs of paths...")
        data.save_all_data()

        print("Calculating averages...")
        data.calc_avg_ast(rotation="n")

        print("Saving plots...")
        data.plot_avg_data(rotation="n", show_plot=True, save_plot=True)
        for a in data.averages:
            a.avg_debug_plot(show_plot=True, save_plot=True)

        print("Consolidating metrics together...")
        results = AstHandAnalyzer(data)

        print("Saving metric data...")
        results.save_data()

        print(f"{h} data generation is complete!")


if __name__ == '__main__':
    home_directory = Path(__file__).parent.absolute()

    run()
