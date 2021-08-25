import pandas as pd
from ast_trial import AstBasicData, AstTrial


class AstTrialRotation(AstTrial):  # TODO: maybe have it inherit AstTrial instead?
    """
    Class which handles cw/ccw trials, separating because we would appreciate the nuance
    """

    def __init__(self, data, rotation_label):
        self.poses = data  # TODO: placeholder for later

        self.trial_translation = 'n'
        self.trial_rotation = rotation_label

        # target path is (0,0) and target rotation should be max rotation value
        # TODO: will need to overwrite the generate target line and rotation methods

        # TODO: assess path labels will have to change

        # TODO: plot trial needs to be filled in... and check plot_orientations

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

        def _condition_df(self, df, norm_data=True):
            """
            Data conditioning procedure used to:
            0) Make columns of the dataframe numeric (they aren't by default), makes dataframe header after the fact to avoid errors with apply function
            1) convert translational data from meters to mm
            2) normalize translational data by hand span/depth
            3) remove extreme outlier values in data
            """
            # df_numeric = df.apply(pd.to_numeric)
            # df = df.set_index("frame")

            # df_numeric.columns = ["pitch", "rmag", "roll", "tmag", "x", "y", "yaw", "z"]
            # TODO: is there a way I can make this directly hit each column without worrying about the order?
            # convert m to mm in translational data
            df = df * [1., 1., 1., 1000., 1000., 1000., 1., 1000.]

            if norm_data:
                # normalize translational data by hand span
                df = df / [1., 1., 1.,  # orientation data
                           1.,  # translational magnitude, don't use
                           self.hand.span,  # x
                           self.hand.depth,  # y
                           1.,  # yaw
                           1.]  # z - doesn't matter
                df = df.round(4)

            # occasionally get an outlier value (probably from vision algorithm), I filter them out here
            # inlier_df = self._remove_outliers(df, ["x", "y", "rmag"])
            if len(df) > 10:  # for some trials with movement, this destroys the data. 10 is arbitrary value that works
                for col in ["x", "y", "rmag"]:
                    # see: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
                    # q_low = df_to_fix[col].quantile(0.01)
                    q_hi = df[col].quantile(0.98)  # determined empirically

                    df = df[(df[col] < q_hi)]  # this has got to be the problem line

            return df.round(4)

        def is_ast_trial(self):
            return False

        def is_avg_trial(self):
            return False

        def is_rot_trial(self):
            return True

        def generate_target_line(self, n_samples=100, no_norm=0):
            pass

        def generate_target_rot(self, n_samples=50):
            pass

        def plot_trial(self):
            pass

        def assess_path_labels(self, no_mvt_threshold=0.1, init_threshold=0.05, init_num_pts=10, dev_perc_threshold=0.10):
            pass

        def update_all_metrics(self, use_filtered=True, redo_target_line=False):
            pass

if __name__ == '__main__':
    test = AstTrialRotation(file_name="sub1_2v2_n_cw_1.csv", do_metrics=True, norm_data=True)
    print(f"name: {test.generate_name()}")
    print(f"tot dist: {test.total_distance}")
    print(f"path labels: {test.path_labels}")
    print(f"metrics: {test.metrics}")

    test.moving_average(window_size=10)
    test.plot_trial(use_filtered=False, provide_notes=True)
