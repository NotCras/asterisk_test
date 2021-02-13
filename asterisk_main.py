from pathlib import Path
import asterisk_hand_data as h
from asterisk_hand_data import AsteriskHandData
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
    subjects = ["sub1", "sub2"]
    hand_names = ["basic", "m2stiff", "m2active", "2v2", "3v3", "2v3", "barrett", "modelvf"]

    failed_files = []
    for h in hand_names:
        print(f"Running: {h}, {subjects}")
        input("Please press <ENTER> to continue")  # added this for debugging by hand

        data = AsteriskHandData(subjects, h)
        data.filter_data()
        data.save_data()
        print("   ")


if __name__ == '__main__':

    home_directory = Path(__file__).parent.absolute()
    run()
