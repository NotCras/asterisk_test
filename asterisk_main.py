from pathlib import Path

def run():
    """
    Handles running an asterisk study
    1) imports all specified data
    2) filters everything with a 15 sample moving average
    3) averages all similar trials
    4) produces a final averaged plot for each hand, as well as by subject
    5) writes salient data tp external files (averaged asterisk data, final statistics, generated plots)
    """
    pass

if __name__ == '__main__':

    home_directory = Path(__file__).parent.absolute()
