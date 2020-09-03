import os
from pathlib import Path
import matplotlib.pyplot as pl

if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    subject_name = input("Enter which subject you want to process: ")
    hand = input("Enter which hand you want to process: ")
    dir_label = input("Enter which direction you want to process: ")
    trial_type = input("Enter what kind of asterisk you are processing: ")
    trial_num = input("Enter which trial number you want to process: ")
    
    data_name = subject_name + "_" + hand + "_" + dir_label + "_" + trial_type + "_" + trial_num

    file_dir = "viz/" + data_name + "/"
    os.chdir(file_dir)

    files = [f for f in os.listdir('.') if f[-3:] == 'jpg']
    files.sort()
    
    img = None
    for f in files:
        im = pl.imread(f)

        if img is None:
            img = pl.imshow(im)
        else:
            img.set_data(im)

        pl.pause(.01)
        pl.draw()

    print("Completed running image data.")
