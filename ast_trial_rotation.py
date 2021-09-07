import pdb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from trial_labelling import AsteriskLabelling as al
from metric_calculation import AstMetrics as am
from ast_trial import AstBasicData, AstTrial


class AstTrialRotation(AstTrial):
    """
    Class which handles cw/ccw trials, separating because we would appreciate the nuance
    """

    def __init__(self, file_name, data=None, # subject_label=None, rotation_label=None, number_label=None,
                 controller_label=None, do_metrics=True, norm_data=True, condition_data=True):

        self.total_distance = 0  # This will be the max rotation value of the trial
        super().__init__(file_name=file_name, data=data, controller_label=controller_label, do_metrics=False,
                         condition_data=condition_data, norm_data=norm_data)

        # TODO: will need to revisit how metrics are calculated

    def _read_file(self, file_name, folder="aruco_data/", norm_data=True, condition_data=True):
        """
        Function to read file and save relevant data in the object
        :param file_name: name of file to read in
        :param folder: name of folder to read file from. Defaults csv folder
        """
        total_path = f"{folder}{file_name}"
        try:
            # print(f"Reading file: {total_path}")
            df = pd.read_csv(total_path, skip_blank_lines=True)
            df = df.set_index("frame")
        except Exception as e:  # TODO: add more specific except clauses
            # print(e)
            print(f"{total_path} has failed to read csv")
            return None

        if condition_data:
            try:
                # print(f"Now at data conditioning.")
                df = self._condition_df(df, norm_data=norm_data)
            except Exception as e:
                # print(e)
                print(f"{total_path} has failed at data conditioning. There's a problem with the data.")
                return None

        return df

    def add_data_by_file(self, file_name, norm_data=True, handinfo_name=None, do_metrics=True, condition_data=True):
        """
        Add object path data as a file. By default, will run data through conditioning function
        """
        # Data will not be filtered in this step
        path_df = self._read_file(file_name, condition_data=condition_data, norm_data=norm_data)

        self.poses = path_df[["x", "y", "rmag"]]

        # self.target_line = self.generate_target_line(100)  # 100 samples
        self.target_line, self.total_distance = self.generate_target_rot()

        self.assess_path_labels()
        print(self.path_labels)

        if do_metrics and self.poses is not None and "no_mvt" not in self.path_labels:
            self.update_all_metrics()

    def add_data_by_df(self, path_df, condition_df=True, do_metrics=True, norm_data=True):
        """
        Add object path data as a dataframe. By default, will run dataframe through conditioning function
        """
        path_df = path_df.set_index("frame")

        if condition_df:
            data = self._condition_df(path_df, norm_data)
        else:
            data = path_df

        self.poses = data[["x", "y", "rmag"]]

        # self.target_line = self.generate_target_line(100)  # 100 samples
        self.target_line, self.total_distance = self.generate_target_rot()

        self.assess_path_labels()

        if do_metrics and self.poses is not None and "no_mvt" not in self.path_labels:
            self.update_all_metrics()

    def is_ast_trial(self):
        return False

    def is_avg_trial(self):
        return False

    def is_rot_trial(self):
        return True

    def generate_target_line(self, n_samples=100, no_norm=0):
        return np.zeros(2)  # 2 zeroes for (x, y)

    def generate_target_rot(self, n_samples=50):
        # make 3d line with rot val from each row in the rmag position, and with 0,0 for each translation
        rotations = self.poses["rmag"]
        max_rot = max(rotations)

        target_df = pd.DataFrame(columns=["x", "y", "rmag"])  # in other object its a list
        len_trial = len(rotations)
        xys = np.zeros((1, len_trial))

        target_df["x"] = xys[0]
        target_df["y"] = xys[0]
        target_df["rmag"] = rotations

        final_target_ln = target_df.dropna().to_numpy()

        return final_target_ln, max_rot

    def plot_trial(self, use_filtered=True, show_plot=True, save_plot=False, provide_notes=False, angle_interval=None):
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        rotation_val = self.total_distance
        data = [rotation_val, 360 - rotation_val]

        # rounds value to integer value for cleaner labelling on plot
        plot_labels = [f"{str(int(rotation_val))}{chr(176)}", ""]  # chr(176) gives us the degrees sign

        if self.trial_rotation == "cw":
            rot_color = "crimson"
            counter_clock = False
            angle_plotted = 90
        elif self.trial_rotation == "ccw":
            rot_color = "royalblue"
            counter_clock = True
            angle_plotted = 90
        else:
            raise AttributeError("Using an AstTrialRotation object incorrectly. Not a cw/ccw trial.")

        colors = [rot_color, "whitesmoke"]

        ax.pie(data, colors=colors, labels=plot_labels, labeldistance=0.8,
               startangle=angle_plotted, counterclock=counter_clock,  # depends on cw or ccw
               textprops=dict(color="whitesmoke", size=11, weight="bold",
                              rotation_mode='anchor', va='center', ha='center'
                              ))

        # draw circle by drawing a white circle on top of center
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        plt.title(f"Plot: {self.generate_plot_title()}, Max Rotation")

        # translation of the trial
        line_x, line_y, _ = self.get_poses(use_filtered=use_filtered)
        plt.plot(line_x, line_y)

        # plot target lines  # TODO: clean up below, move into function in data_plotting?
        target_line_dist = 0.5

        line_a_x = [0, 0]
        line_a_y = [0, target_line_dist]

        line_e_x = [0, 0]
        line_e_y = [0, -target_line_dist]

        line_c_x = [0, target_line_dist]
        line_c_y = [0, 0]

        line_g_x = [0, -target_line_dist]
        line_g_y = [0, 0]

        lines_x = [line_a_x, line_c_x, line_e_x, line_g_x]
        lines_y = [line_a_y, line_c_y, line_e_y, line_g_y]

        # plot the translation that occurred during the trial inside the donut hole
        for x, y in zip(lines_x, lines_y):
            plt.plot(x, y, color='lightsteelblue', linestyle="--")

        # show circle for deviation limits
        angle = np.linspace(0, 2 * np.pi, 150)
        radius = 0.1
        x_lim = radius * np.cos(angle)
        y_lim = radius * np.sin(angle)
        plt.plot(x_lim, y_lim, color='red', linestyle=(0, (1, 3)) )

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        if provide_notes:
            self._plot_notes()

        if save_plot:
            plt.savefig(f"results/pics/plot_{self.generate_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        if show_plot:
            plt.show()

    def assess_path_labels(self, no_mvt_threshold=10, init_threshold=0.05, init_num_pts=10, dev_perc_threshold=0.10):
        # check whether total distance is an acceptable distance to consider it actually movement
        # this one is based on rotation value
        if self.total_distance < no_mvt_threshold:  # TODO: should this be arc len based? Or incorporate arc len too?
            self.path_labels.append("no_mvt")
            print(f"No movement detected in {self.generate_name()}. Skipping metric calculation.")

        # check that data starts near center
        if not al.assess_initial_position(self, threshold=init_threshold, to_check=init_num_pts):
            self.path_labels.append("not centered")
            print(f"Data for {self.generate_name()} failed, did not start at center.")

        deviated, dev_perc = al.assess_path_deviation_with_rotation(self)

        if deviated and dev_perc > dev_perc_threshold:
            self.path_labels.append("major deviation")
            print(f"Detected major deviation in {self.generate_name()} at {dev_perc}%. Labelled trial.")
        elif deviated:
            self.path_labels.append("deviation")
            print(f"Detected minor deviation in {self.generate_name()} at {dev_perc}%. Labelled trial.")

        # TODO: revisit this label for rotations... do we even want this?
        # mvt_observations = al.assess_path_movement(self)
        #
        # if "backtracking" in mvt_observations:
        #     self.path_labels.append("backtracking")
        #
        # if "shuttling" in mvt_observations:
        #     self.path_labels.append("shuttling")

        return self.path_labels

    def update_all_metrics(self, use_filtered=True, redo_target_line=False):
        """
        Updates all metric values on the object.
        """
        if redo_target_line:
            self.target_line, self.total_distance = self.generate_target_rot()

        # TODO: figure out naming here... we only get one frechet distance value for rot trials
        translation_fd, rotation_fd = am.calc_frechet_distance(self)
        # TODO: translation_fd becomes max translation magnitude error from zero, rotation_fd will be full line?
        # fd = am.calc_frechet_distance_all(self)

        #  TODO: might have a problem here, rotation might heavily outweigh translation
        mvt_efficiency, arc_len = am.calc_mvt_efficiency(self, with_rot=True)

        max_error = am.calc_rot_max_error(self, arc_len)

        # TODO: need to revisit the following two function calls because they are a little janky with rotation only
        area_btwn = am.calc_area_btwn_curves(self)

        # this one is particularly troublesome
        max_area_region, max_area_loc = am.calc_max_area_region(self)

        # TODO: Make getters for each metric - can also return none if its not calculated
        metric_dict = {"trial": self.generate_name(), "dist": self.total_distance,
                       "t_fd": translation_fd, "r_fd": rotation_fd,  # "fd": fd
                       "max_err": max_error, "mvt_eff": mvt_efficiency, "arc_len": arc_len,
                       "area_btwn": area_btwn, "max_a_reg": max_area_region, "max_a_loc": max_area_loc}

        self.metrics = pd.Series(metric_dict)
        return self.metrics


if __name__ == '__main__':
    test = AstTrialRotation(file_name="sub1_2v2_n_ccw_2.csv", do_metrics=False, norm_data=True)
    print(f"name: {test.generate_name()}")
    print(f"tot dist: {test.total_distance}")
    print(f"path labels: {test.path_labels}")
    print(f"metrics: {test.metrics}")

    test.moving_average(window_size=10)
    test.plot_trial(use_filtered=False, provide_notes=True)
