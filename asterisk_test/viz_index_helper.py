from pathlib import Path
from matplotlib.widgets import Slider, Button
import data_manager as datamanager
import pandas as pd
import matplotlib.pyplot as plt
import os


class ArucoIndices:
    """
    Handles data indices, beginning and ending indices
    """

    @staticmethod
    def get_indices(id, file_name=None):
        """
        Gets the beginning and ending indices of the
        :param id:
        :param file_name:
        :return:
        """
        if file_name is not None:
            table = pd.read_csv(file_name)
        else:
            # table = pd.read_csv("viz_data_indices.csv")
            table = pd.read_csv("viz_data_indices_main.csv")

        table = table.set_index("id")
        try:
            indices = table.loc[id]

        except Exception as e:
            print("Could not find the index.")
            print(e)
            raise IndexError("Could not find the correct indices")

        return int(indices["begin_idx"]), int(indices["end_idx"])

    @staticmethod
    def find_indices(folder_path):
        """
        Helper to get the data indices for the start and end of a trial
        """
        home = Path(__file__).parent.absolute()
        os.chdir(folder_path)
        files = [f for f in os.listdir('.') if f[-3:] == 'jpg']
        files.sort()

        # get start index
        start = ArucoIndices._slider_window(files, title="START")

        # get end index
        end = ArucoIndices._slider_window(files, title="END", init_val=int(start+1))

        print(folder_path)
        print(f"start: {start}, end: {end}")

        os.chdir(home)

        if end <= start:
            raise ValueError("starting index cannot be greater than ending index!")

        return start, end

        # TODO: add saving functionality, add these values to the stored index values?

    @staticmethod
    def _slider_window(list_of_files, title, init_val=0):
        """
        Generates a matplotlib window with a slider that changes which image is shown
        """
        num_files = len(list_of_files)

        # setup
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        # show initial image
        image_id = list_of_files[0]
        image = plt.imread(image_id)
        plt.imshow(image)
        plt.draw()
        plt.title(f"Get {title} index!")

        # setup slider
        slider_bkd_color = "lightgoldenrodyellow"
        axpos = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=slider_bkd_color)
        # allowed_positions = np.linspace(0, 1, num_files).tolist()
        image_pos = Slider(
            ax=axpos,
            label="Image Number",
            valmin=0,
            valmax=num_files-1,
            valinit=init_val,
            valstep=1
            #init_color="none"
        )

        def _slider_update(val):
            print(f"val: {val}")
            image_id = list_of_files[int(val)]
            image = plt.imread(image_id)
            ax.imshow(image)
            ax.draw()  # if I use plt.draw() it will draw the image in the button's space

        image_pos.on_changed(_slider_update)

        ax_exit = plt.axes([0.8, 0.025, 0.13, 0.05])
        ax_left = plt.axes([0.3, 0.025, 0.1, 0.05])
        ax_right = plt.axes([0.5, 0.025, 0.1, 0.05])

        button_exit = Button(ax_exit, 'Got Index!', color=slider_bkd_color, hovercolor='0.975')
        button_left = Button(ax_left, '<-', color=slider_bkd_color, hovercolor='0.975')
        button_right = Button(ax_right, '->', color=slider_bkd_color, hovercolor='0.975')

        def _button_exit(val):
            print(f"start index: {image_pos.val}")
            plt.close()

        def _button_left(val):
            i = image_pos.val
            if i <= 0:
                i = 0
            else:
                i = i - 1

            image_pos.set_val(i)

        def _button_right(val):
            i = image_pos.val
            if i >= num_files-1:
                i = num_files-1
            else:
                i = i + 1

            image_pos.set_val(i)

        button_exit.on_clicked(_button_exit)
        button_left.on_clicked(_button_left)
        button_right.on_clicked(_button_right)

        plt.show()
        return image_pos.val


if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    ans = datamanager.smart_input("Enter a function", "mode", ["1", "2"])
    subject = datamanager.smart_input("Enter subject name: ", "subjects")
    hand = datamanager.smart_input("Enter name of hand: ", "hands")
    translation = datamanager.smart_input("Enter type of translation: ", "translations_w_n")

    if translation == "n":
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotations_n_trans")
    else:
        rotation = datamanager.smart_input("Enter type of rotation: ", "rotation_combos")

    trial_num = datamanager.smart_input("Enter trial number: ", "numbers")

    file_name = f"{subject}_{hand}_{translation}_{rotation}_{trial_num}"

    if ans=="1":
        folder_path = f"{file_name}/"

        try:
            b_idx, e_idx = ArucoIndices.get_indices(file_name)
        except:
            e_idx = None
            b_idx = 0

        print(f"b: {b_idx}, e: {e_idx}")

    elif ans=="2":
        folder_path = f"viz/{file_name}/"

        s_i, e_i = ArucoIndices.find_indices(folder_path)