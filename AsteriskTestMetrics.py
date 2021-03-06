#!/usr/bin/env python3

import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv
from pathlib import Path
import os

from AsteriskTestTypes import generate_fname, AsteriskTestTypes


class Pose2D:
    def __init__(self, in_x=0, in_y=0, in_theta=0):
        """ Pose in 2 dimensions
        :param in_x x location
        :param in_y y location
        :param in_theta rotation angle, in degrees"""

        self.x = in_x
        self.y = in_y
        self.theta = in_theta

    def distance(self, pose, scl_ratio=(0.5, 0.5)):
        """Difference between self and another pose
        :type pose: Pose2D
        :param pose another Pose2D
        :param scl_ratio - how much to scale distance and rotation error by
        :returns a number"""

        # Standard euclidean distance, scaled by sin(45)
        dist_trans = sqrt((self.x - pose.x) ** 2 + (self.y - pose.y) ** 2) / sin(pi/4)
        # set up 20 deg of error as approximately 1 unit of translation error
        ang_rot = abs(self.theta - pose.theta)
        if ang_rot > 180:
            ang_rot -= 180
        dist_rot = (ang_rot / 45.0)

        # return the average
        return scl_ratio[0] * dist_trans + scl_ratio[1] * dist_rot

    def lin_interp(self, pose, t):
        """ linearly interpolate the pose values
        :param pose pose to interpoloate to Pose2D
        :param t a number between 0 and 1
        :return the interpolated pose Pose2D"""
        return Pose2D(self.x + t * pose.x, self.y + t * pose.y, self.theta + t * pose.theta)

    def __str__(self):
        return "x : {0:.2f}, y : {1:.2f}, theta : {2}".format(self.x, self.y, self.theta)


class AsteriskTestResults(AsteriskTestTypes):
    status_values = {"Successful", "Unsuccessful", "Not_tried"}
    def __init__(self, name, in_obj_poses: np.array, in_target_path: [Pose2D]):
        """distances, angles, index in test
        :param name - trial file name"""
        self.test_name = name

        # Actual distances
        self.dist_target = nan
        self.dist_frechet = nan
        self.dist_along_translation = nan
        self.dist_along_rotation = nan

        # Data for Frechet distance
        self.end_target_index = -1
        self.target_indices = []

        # Save data
        self.obj_poses = in_obj_poses.copy()  # Copy because re-written each time
        self.target_path = in_target_path # These should be stable

    def __str__(self):
        """Print results"""
        ret_str = super().__str__()
        if self.end_target_index == -1:
            ret_str = ret_str + ", no result"
        if self.dist_target is not nan:
            ret_str = ret_str + " Target: {0:0.3f}".format(self.dist_target)
        if self.dist_along_translation is not nan:
            ret_str = ret_str + " D Trans: {0:0.3f}".format(self.dist_along_translation)
        if self.dist_along_rotation is not nan:
            ret_str = ret_str + " D Rot: {0:0.3f}".format(self.dist_along_rotation)
        if self.dist_frechet is not nan:
            ret_str = ret_str + " D Frec: {0:0.3f}".format(self.dist_frechet)
        return ret_str

    @staticmethod
    def write_header_data(f):
        """ Header row for csv file """
        col_names = ["Name", "Type", "TranslationName", "RotationName", "TwistName"]
        col_names.extend(["DistToTarget", "DistAlongTranslation", "DistAlongRotation", "FrechetDistance"])
        col_names.extend(["LastIndex", "Indices"])
        f.writerow(col_names)

    def write_data(self, f):
        """ Write out data to a csv file
        :param f - csv file writer"""

        row_data = []
        row_data.append(self.test_name)
        row_data.append(self.get_test_name())
        row_data.append(self.get_translation_name())
        row_data.append(self.get_rotation_name())
        row_data.append(self.get_twist_name())
        row_data.append(self.dist_target)
        row_data.append(self.dist_along_translation)
        row_data.append(self.dist_along_rotation)
        row_data.append(self.dist_frechet)
        row_data.append(self.end_target_index)
        for i in self.target_indices:
            row_data.append(f"{i}")

        f.writerow(row_data)


class AsteriskTestMetrics2D:
    metric_names = {"Distance_along, Distance_target, Frechet_distance"}
    translation_angles = linspace(90, 90-360, 8, endpoint=False)

    def __init__(self, n_samples=15):
        """
        :param n_samples number of samples in target path"""
        self.target_paths = {}
        self.test_results = []
        self._add_target_paths(n_samples)
        self.reset_test_results()
        self.test_name = ""

    def reset_test_results(self):
        """Zero out the results"""

        self.test_results = []

    def _add_target_paths(self, n_samples):
        """Create ideal paths for each movement type
        :param n_samples number of samples to use on path"""

        target_translation_paths = []
        target_rotation_translation_paths = [[], []]
        divs = linspace(0, 1, n_samples, endpoint=True)
        for i, a in enumerate(self.translation_angles):
            x = cos(pi * a / 180)
            y = sin(pi * a / 180)
            target_translation_paths.append([])
            target_rotation_translation_paths[0].append([])
            target_rotation_translation_paths[1].append([])
            for d in divs:
                target_translation_paths[-1].append(Pose2D(x * d, y * d, 0))
                target_rotation_translation_paths[0][-1].append(Pose2D(x * d, y * d, AsteriskTestTypes.twist_directions["Clockwise"]))
                target_rotation_translation_paths[1][-1].append(Pose2D(x * d, y * d, AsteriskTestTypes.twist_directions["Counterclockwise"]))

        target_rotation_paths = [[], []]
        for d in divs:
            target_rotation_paths[0].append(Pose2D(0, 0, d * AsteriskTestTypes.rotation_directions["Clockwise"]))
            target_rotation_paths[1].append(Pose2D(0, 0, d * AsteriskTestTypes.rotation_directions["Counterclockwise"]))

        self.target_paths["Translation"] = target_translation_paths
        self.target_paths["Rotation"] = target_rotation_paths
        self.target_paths["Twist_translation"] = target_rotation_translation_paths

    @staticmethod
    def _narrow_target(obj_pose, target_poses, scl_ratio=(0.5, 0.5)) -> int:
        """ narrown down the closest point on the target poses
        :param obj_pose last object pose Pose2D
        :param target_poses [Pose2D;
        :param scl_ratio - how much to scale distance and rotation error by
        :returns target_i the index of the best match """

        dists_targets = [obj_pose.distance(p, scl_ratio) for p in target_poses]
        i_target = dists_targets.index(min(dists_targets))

        return i_target

    @staticmethod
    def _frechet_dist(poses_obj, i_target, target_poses, scl_ratio=(0.5, 0.5)) ->(int, float):
        """ Implement Frechet distance
        :param poses_obj all the object poses np.array
        :param i_target the closest point in target_poses
        :param target_poses [Pose2D];
        :param scl_ratio - how much to scale distance and rotation error by
        :returns max of the min distance between target_poses and obj_poses """
        # https://towardsdatascience.com/fast-discrete-fr%C3%A9chet-distance-d6b422a8fb77

        # Length of target curve
        n_target = min(i_target+1, len(target_poses))
        # Length of object path
        n_object_path = poses_obj.shape[1]

        # Matrix to store data in as we calculate Frechet distance
        # Entry i,j has the cost of pairing i with j, assuming best pairings up to that point
        ca = np.zeros((n_target, n_object_path), dtype=np.float64)
        ds = np.zeros((n_target, n_object_path), dtype=np.float64)
        dsum = np.zeros((n_target, n_object_path), dtype=np.float64)
        imatch = np.zeros((n_target, n_object_path), dtype=np.int)

        print("Frechet debug: target {0}, n {1}".format(i_target, n_object_path))

        # Top left corner
        ca[0, 0] = target_poses[0].distance(Pose2D(poses_obj[0, 0], poses_obj[1, 0], poses_obj[2, 0]), scl_ratio)
        ds[0, 0] = ca[0, 0]
        dsum[0, 0] = ds[0, 0]
        imatch[0, 0] = 0  # Match the first target pose to the first object pose
        target_index = [i for i in range(0, n_target)]

        # Fill in left column ...
        for i_t in range(1, n_target):
            ds[i_t, 0] = target_poses[i_t].distance(Pose2D(poses_obj[0, 0], poses_obj[1, 0], poses_obj[2, 0]), scl_ratio)
            ca[i_t, 0] = max(ca[i_t-1, 0], ds[i_t, 0])
            dsum[i_t, 0] = dsum[i_t - 1, 0] + ds[i_t, 0]
            imatch[i_t, 0] = 0  # Match the ith target pose to the first object pose

        # ... and top row
        for i_p in range(1, n_object_path):
            ds[0, i_p] = target_poses[0].distance(Pose2D(poses_obj[0, i_p], poses_obj[1, i_p], poses_obj[2, i_p]), scl_ratio)
            ca[0, i_p] = max(ca[0, i_p - 1], ds[0, i_p])
            if ds[0, i_p] < dsum[0, i_p - 1]:
                imatch[0, i_p] = i_p  # Match the first target pose to this object pose
                dsum[0, i_p] = ds[0, i_p]
            else:
                imatch[0, i_p] = imatch[0, i_p-1] # Match to an earlier pose
                dsum[0, i_p] = dsum[0, i_p-1]

        # Remaining matrix
        for i_t in range(1, n_target):
            tp = target_poses[i_t]
            for i_p in range(1, n_object_path):
                ds[i_t, i_p] = tp.distance(Pose2D(poses_obj[0, i_p], poses_obj[1, i_p], poses_obj[2, i_p]), scl_ratio)
                ca[i_t, i_p] = max(min(ca[i_t - 1, i_p],
                                       ca[i_t - 1, i_p - 1],
                                       ca[i_t, i_p - 1]),
                                   ds[i_t, i_p])
                # Compare using this match with the best match upper row so far
                # to best match found so far
                if ds[i_t, i_p] + dsum[i_t-1, i_p] < dsum[i_t, i_p-1]:
                    imatch[i_t, i_p] = i_p
                    dsum[i_t, i_p] = ds[i_t, i_p] + dsum[i_t-1, i_p]
                else:
                    dsum[i_t, i_p] = dsum[i_t, i_p-1]    # Keep the old match
                    imatch[i_t, i_p] = imatch[i_t, i_p-1]

        # initialize with minimum value match - allows backtracking
        target_index = []
        v_min = np.amin(ds, axis=1)
        for r in range(0, n_target):
            indx = np.where(ds[r, :] == v_min[r])
            target_index.append(indx[0][0])

        b_is_ok = True
        for i in range(0, n_target-1):
            if target_index[i+1] < target_index[i]:
                b_is_ok = False
                print("Frechet: Found array not sorted")

        # Could just do this, but leaving be for a moment to ensure working
        if b_is_ok == False:
            for i_t in range(0, n_target):
                target_index[i_t] = imatch[i_t, n_object_path-1]

        return ca[n_target-1, n_object_path-1], target_index

    def add_translation_test(self, name, in_which, poses_obj):
        """Add the translation test
        :param name: str - name of test, eg, hand type
        :param in_which is 0..7, which angle in the asterisk
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        target_path = self.target_paths["Translation"][in_which]
        target_pose = target_path[-1]

        # Setup test results
        ret_dists = AsteriskTestResults(name, poses_obj, target_path)
        ret_dists.set_translation_test(in_which)

        # Check that we're in at least roughly the right ballpark for the end pose
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        n_total = poses_obj.shape[1]
        last_pose_angle = 180.0 * arctan2(last_pose_obj.y, last_pose_obj.x) / pi
        expected_angle = self.translation_angles[in_which]
        if last_pose_angle - expected_angle > 180:
            last_pose_angle -= 360
        elif expected_angle - last_pose_angle > 180:
            last_pose_angle += 360

        if abs(last_pose_angle - expected_angle) > 65:
            print("Warning: Translation {} detected bad last pose {}, expected {}".format(in_which, last_pose_angle,
                                                                                          self.translation_angles[
                                                                                              in_which]))

        # Use mostly translation for distance
        scl_ratio = (1- 0.01, 0.01)
        ret_dists.dist_along_translation = sqrt(max([poses_obj[0, i_p]**2 + poses_obj[1, i_p]**2 for i_p in range(0, n_total)]))
        ret_dists.end_target_index = self._narrow_target(last_pose_obj, target_path, scl_ratio)

        ret_dists.dist_target = last_pose_obj.distance(target_pose, scl_ratio)

        if ret_dists.end_target_index == 0:
            print("Warning: Closest pose was first pose")
            ret_dists.end_target_index += 1

        ret_dists.dist_frechet, ret_dists.target_indices = self._frechet_dist(poses_obj,
                                                                              ret_dists.end_target_index,
                                                                              target_path,
                                                                              scl_ratio)
        self.test_results.append(ret_dists)
        return ret_dists

    def add_rotation_test(self, name, in_which, poses_obj):
        """Add the translation test
        :param in_which is Clockwise or Counterclockwise
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        target_path = self.target_paths["Rotation"][in_which]
        target_pose = target_path[-1]

        # Setup test results
        ret_dists = AsteriskTestResults(name, poses_obj, target_path)
        ret_dists.set_rotation_test(in_which)

        # Check that we're in at least roughly the right ballpark for the end pose
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        dist_from_center = sqrt(last_pose_obj.x**2 + last_pose_obj.y**2)
        if dist_from_center > 0.2:
            print("Warning: rotation test {} had big offset {}".format(in_which, dist_from_center))

        # Use mostly rotation error for distance
        scl_ratio = (0.01, 1 - 0.01)

        # Calculate distances
        ret_dists.dist_along_rotation = abs(last_pose_obj.theta) / AsteriskTestTypes.rotation_directions["Counterclockwise"]
        ret_dists.end_target_index = self._narrow_target(last_pose_obj, target_path, scl_ratio)

        ret_dists.dist_target = last_pose_obj.distance(target_pose)

        if ret_dists.end_target_index == 0:
            print("Warning: Closest pose was first pose")
            ret_dists.end_target_index += 1

        ret_dists.dist_frechet, ret_dists.target_indices = self._frechet_dist(poses_obj,
                                                                              ret_dists.end_target_index,
                                                                              target_path,
                                                                              scl_ratio)

        self.test_results.append(ret_dists)
        return ret_dists

    def add_twist_translation_test(self, name, in_which_rot, in_which_trans, poses_obj):
        """Add the translation test
        :param in_which_rot is 0 or 1, clockwise or counter
        :param in_which_trans is 0..7, which angle in the asterisk
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        target_path = self.target_paths["Twist_translation"][in_which_rot][in_which_trans]
        target_pose = target_path[-1]

        # Setup test results
        ret_dists = AsteriskTestResults(name, poses_obj, target_path)
        ret_dists.set_twist_translation_test(in_which_trans, in_which_rot)

        # Check that we're in at least roughly the right ballpark for the end pose
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        n_total = poses_obj.shape[1]

        last_pose_angle = 180.0 * arctan2(last_pose_obj.y, last_pose_obj.x) / pi
        expected_angle = self.translation_angles[in_which_trans]
        if last_pose_angle - expected_angle > 180:
            last_pose_angle -= 360
        elif expected_angle - last_pose_angle > 180:
            last_pose_angle += 360

        if abs(last_pose_angle - expected_angle) > 65:
            print("Warning: Translation {} detected bad last pose {}, expected {}".format(in_which_trans, last_pose_angle,
                                                                                          self.translation_angles[
                                                                                              in_which_trans]))

        # Use mostly translation for distance
        scl_ratio = (1 - 0.01, 0.01)

        ret_dists.dist_along_translation = sqrt(max([poses_obj[0, i_p]**2 + poses_obj[1, i_p]**2 for i_p in range(0, n_total)]))
        ret_dists.dist_along_rotation = abs(last_pose_obj.theta) / AsteriskTestTypes.twist_directions["Counterclockwise"]

        ret_dists.end_target_index = self._narrow_target(last_pose_obj, target_path, scl_ratio)

        ret_dists.dist_target = last_pose_obj.distance(target_pose)

        if ret_dists.end_target_index == 0:
            print("Warning: Closest pose was first pose")
            ret_dists.end_target_index += 1

        ret_dists.dist_frechet, ret_dists.target_indices = self._frechet_dist(poses_obj,
                                                                              ret_dists.end_target_index,
                                                                              target_path,
                                                                              scl_ratio)
        ret_dists.target_paths = self.target_paths

        self.test_results.append(ret_dists)
        return ret_dists

    def get_test_results(self, att: AsteriskTestTypes)->[AsteriskTestResults]:
        """Get all the tests of that type
        :param att Asterisk test type
        :returns [AsteriskTestResults] """

        return [res for res in self.test_results if res.is_type(att)]

    def write_test_results(self, fname):
        """ Write the test results out to a csv file
        :param fname: str
        :rtype none"""
        with open(fname, "w") as f:
            csv_f = csv.writer(f, delimiter=',')
            AsteriskTestResults.write_header_data(csv_f)
            for t in self.test_results:
                t.write_data(csv_f)

    def test_translation(self):
        """ Make fake translation data and add it
        :rtype: None
        """
        from numpy.random import uniform

        n_poses = 100
        obj_poses = zeros([3, n_poses])
        noise_x = uniform(-0.1, 0.1, (8, n_poses))
        noise_y = uniform(-0.1, 0.1, (8, n_poses))
        noise_ang = uniform(-5, 5, (8, n_poses))

        print("Testing translations")
        for i_ang, ang in enumerate(self.translation_angles):
            for i_p, div in enumerate(linspace(0, 1, n_poses)):
                obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
                obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
                obj_poses[2, i_p] = noise_ang[i_ang, i_p]

            dists = self.add_translation_test("Test", i_ang, obj_poses)
            print("Translation, angle {0}: {1}".format(ang, dists))

    def test_rotation(self):
        """ Make fake translation data and add it
        :rtype: None """
        from numpy.random import uniform

        n_poses = 50
        obj_poses = zeros([3, n_poses])
        noise_x = uniform(-0.1, 0.1, n_poses)
        noise_y = uniform(-0.1, 0.1, n_poses)
        noise_ang = uniform(-0.5, 0.5, n_poses)

        for i_ang, ang in enumerate(linspace(0, AsteriskTestTypes.rotation_directions["Clockwise"], n_poses)):
            obj_poses[0, i_ang] = noise_x[i_ang]
            obj_poses[1, i_ang] = noise_y[i_ang]
            obj_poses[2, i_ang] = ang + noise_ang[i_ang]

        dists = self.add_rotation_test("Test", 0, obj_poses)
        print("Rotation, angle {0}, distances {1}".format("Clockwise", dists))
        for i_ang, ang in enumerate(linspace(0, AsteriskTestTypes.rotation_directions["Counterclockwise"], n_poses)):
            obj_poses[2, i_ang] = ang + noise_ang[i_ang]
        dists = self.add_rotation_test("Test", 1, obj_poses)
        print("Rotation, angle {0}: {1}".format("Counterclockwise", dists))

    def test_twist_translation(self):
        """ Make fake translation data and add it
        :rtype: None """
        from numpy.random import uniform

        n_poses = 100
        obj_poses = zeros([3, n_poses])
        noise_x = uniform(-0.1, 0.1, (8, n_poses))
        noise_y = uniform(-0.1, 0.1, (8, n_poses))
        noise_ang = uniform(-5, 5, (8, n_poses))

        for i_ang, ang in enumerate(self.translation_angles):
            for i_p, div in enumerate(linspace(0, 1, n_poses)):
                obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
                obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
                obj_poses[2, i_p] = AsteriskTestTypes.twist_directions["Clockwise"] + noise_ang[i_ang, i_p]

            dists = self.add_twist_translation_test("Test", 0, i_ang, obj_poses)
            print("Twist {} Translation, angle {}: {}".format("Clockwise", ang, dists))

        for i_ang, ang in enumerate(self.translation_angles):
            for i_p, div in enumerate(linspace(0, 1, n_poses)):
                obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
                obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
                obj_poses[2, i_p] = AsteriskTestTypes.twist_directions["Counterclockwise"] + noise_ang[i_ang, i_p]

            dists = self.add_twist_translation_test("Test", 1, i_ang, obj_poses)
            print("Twist {} Translation, angle {}: {}".format("Counterclockwise", ang, dists))

    @staticmethod
    def run_tests():
        my_asterisk_tests = AsteriskTestMetrics2D()

        my_asterisk_tests.test_translation()
        my_asterisk_tests.test_rotation()
        my_asterisk_tests.test_twist_translation()

        my_asterisk_tests.write_test_results("test_results.csv")

    def process_file(self, name: str, trial_type: str, ang_name: str, obj_poses):
        """Read the files, compute the metrics
        :param name subject + hand + trial name
        :param trial_type from asterisk_0_prompts, minus, plus, etc
        :param ang_name from asterisk_0_prompts, none or a, b, c etc
        :param obj_poses object poses
        :returns Distances"""

        trial_direction = ord(trial_type[0]) - ord('a')
        if ang_name == "m15":
            dists = self.add_twist_translation_test(name, 1, trial_direction, obj_poses)

        elif ang_name == "p15":
            dists = self.add_twist_translation_test(name, 0, trial_direction, obj_poses)

        elif ang_name == "cw":
            dists = self.add_rotation_test(name, 1, obj_poses)

        elif ang_name == "ccw":
            dists = self.add_rotation_test(name, 0, obj_poses)

        else:
            dists = self.add_translation_test(name, trial_direction, obj_poses)

        return dists

    @staticmethod
    def process_files(subject_name, hand):
        """Read the files, compute the metrics
        # !param dir_name input file name
        :param subject_name name of subject to process
        :param hand name of hand to process
        :return my_tests Array of AsteriskTestMetrics with tests"""

        my_tests = []
        for fname in generate_fname(subject_name, hand):
            _, subject, hand, translation, rotation, end = fname.split("_")
            number, _ = end.split(".")

            try:
                with open(fname, "r") as csvfile:
                    csv_file = csv.reader(csvfile, delimiter=',')
                    obj_poses = []
                    for i, row in enumerate(csv_file):
                        try:
                            obj_poses.append([float(row[1]), float(row[2]), float(row[3])])
                        except:
                            pass

                    obj_poses = np.transpose(np.array(obj_poses))
                    print(f"{fname}\n x {obj_poses[0,-1]} y {obj_poses[1, -1]} t {obj_poses[2, -1]}")

                    hand_name = hand
                    trial_translation = translation
                    trial_rotation = rotation
                    trial_number = int(number)-1
                    name = f"{subject_name}_{hand_name}_Trial{trial_number}"

                    while len(my_tests) <= trial_number:
                        my_tests.append(AsteriskTestMetrics2D())

                    my_tests[trial_number].test_name = f"{subject_name}_{hand_name}"
                    dists = my_tests[trial_number].process_file(name, trial_translation, trial_rotation, obj_poses)

                    print(f"{dists}\n")
            except FileNotFoundError:
                print(f"File not found: {fname}")

        return my_tests

if __name__ == '__main__':
    #AsteriskTestMetrics2D.run_tests()

    subjects = ["sub1", "sub2"]
    hand_names = ["basic", "m2stiff", "m2active", "2v2", "3v3", "2v3", "barrett", "modelvf"]

    home_directory = Path(__file__).parent.absolute()
    file_dir = "filtered/"

    for h in hand_names:
        for s in subjects:
            os.chdir(file_dir)
            subject_name_process = f"f15_{s}"
            hand_process = h
            my_test_results = AsteriskTestMetrics2D.process_files(subject_name_process, hand_process)

            os.chdir(home_directory)

            try:
                for i, t in enumerate(my_test_results):
                    t.write_test_results(f"check_res_{t.test_name}_{i}.csv")
            except Exception as e:
                print(f"{t.test_name}, {e}")
