import numpy as np

class ArucoAutoCrop:
    def __init__(self, apose_obj, only_rotation=False):
        self.pose_obj = apose_obj
        self.trial_data = apose_obj.est_poses

        print("Running autocropper!")
        self.start_i, self.end_i, self.cropped_path_dist, _ = self.auto_crop(only_rotation=only_rotation)
        # TODO: make separate object for ArucoAutoCropRotation?

        self.cropped_poses = self.trial_data.loc[self.start_i:self.end_i]

    def get_cropped_poses(self):
        return self.cropped_poses

    def get_autocrop_indices(self):
        return self.start_i, self.end_i

    def save_poses(self, file_name_overwrite=None):
        """
        Saves pose data as a new csv file
        :param file_name_overwrite: optional parameter, will save as generate_name unless a different name is specified
        """
        if file_name_overwrite is None:
            data_name = self.pose_obj.vision_data.trial_name
            folder = "aruco_data"  # "csv"
            new_file_name = f"{folder}/{data_name}.csv"

        else:
            new_file_name = file_name_overwrite + ".csv"

        self.cropped_poses.to_csv(new_file_name, index=True)
        # print(f"CSV File generated with name: {new_file_name}")

    def yield_index_pairs(self, desired_rotation=None):
        """
        yields pairs of indices to check auto cropping on
        first index goes through all of the indices except the last index in the data
        the second index goes through all of the indices between the first index and the last index
        """
        data_size = len(self.trial_data)
        start_i = 0

        if desired_rotation is not None and (isinstance(desired_rotation, int) or isinstance(desired_rotation, float)):
            # find the first index that achieves the desired rotation in the data set
            # self.trial_data.rmag.eq(15).idxmax()
            i = 0
            for val in self.trial_data.rmag:
                if abs(val-desired_rotation) < 6:  # TODO: revisit this logic
                    start_i = i
                    break

                i += 1

        else:
            start_i = 0  # redundant?

        for i1 in range(start_i, data_size - 1):
            for i2 in range(i1 + 1, data_size):
                yield i1, i2

    def auto_crop(self, only_rotation=False):
        """
        Crops an image trial automatically, but finding the largest distance travelled in the smallest range of index
        :param df_data
        :return:
        """
        trial_length = len(self.trial_data)
        c_max_is = (1, trial_length)

        c_max_dist = 0
        c_min_di = trial_length

        for i1, i2 in self.yield_index_pairs():
            # print(f"Attempting index pair: {i1}, {i2}")
            d1 = self.trial_data.iloc[i1]
            d2 = self.trial_data.iloc[i2]

            # the distance between the sampled points
            if only_rotation:
                i_dist = np.sqrt((d2['rmag']-d1['rmag'])**2)
            else:
                i_dist = np.sqrt((d2['x']-d1['x'])**2 + (d2['y']-d1['y'])**2)  # + (d2['rmag'] - d1['rmag'])**2)
            d_i = i2 - i1

            val_is_close = abs(c_max_dist - i_dist) <= c_max_dist * 0.01

            # now check for...
            if i_dist >= c_max_dist:  # if we record a greater distance than before...
                # print(f"max dist is larger => max:{c_max_dist}, current:{i_dist}")
                # print("overwriting max and index values.")
                # print(f"i1 data: x:{d1['x']}, y:{d1['y']}, t:{d1['rmag']}")
                # print(f"i2 data: x:{d2['x']}, y:{d2['y']}, t:{d2['rmag']}")
                # print(f"total dist: {i_dist}, d_i: {d_i}")
                c_max_dist = i_dist
                c_min_di = d_i
                c_max_is = (i1, i2)  # record which indices we are saving

            if val_is_close:
                # print("value is within 1% of c_max_dist")

                if d_i <= c_min_di:  # is the space between the indices smaller than before?
                    # print(f"d_i is smaller than previous min => min:{c_min_di}, current:{d_i}")
                    # print(f"but the distances are within 1% => max:{c_max_dist}, current:{i_dist} ")
                    # print(f"overwriting max and index values.")
                    c_max_dist = i_dist
                    c_min_di = d_i

                    c_max_is = (i1, i2)  # record which indices we are saving

            # print("  ")
        print(f"cropped indices => start:{c_max_is[0]} | end:{c_max_is[1]}")

        return c_max_is[0], c_max_is[1], c_max_dist, c_min_di