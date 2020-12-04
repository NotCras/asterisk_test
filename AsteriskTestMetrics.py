#!/usr/bin/env python3
from typing import List, Any, Tuple

import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv
from asterisk_0_prompts import generate_fname, dir_options_no_rot, type_options


class Pose2D:
    def __init__(self, in_x=0, in_y=0, in_theta=0):
        """ Pose in 2 dimensions
        :param in_x x location
        :param in_y y location
        :param in_theta rotation angle, in degrees"""

        self.x = in_x
        self.y = in_y
        self.theta = in_theta

    def distance(self, pose):
        """Difference between self and another pose
        :type pose: Pose2D
        :param pose another Pose2D
        :returns a number"""

        # Standard euclidean distance
        dist_trans = sqrt((self.x - pose.x) ** 2 + (self.y - pose.y) ** 2)
        # set up 20 deg of error as approximately 1 unit of translation error
        ang_rot = abs(self.theta - pose.theta)
        if ang_rot > 180:
            ang_rot -= 180
        dist_rot = (ang_rot / 20.0)

        # return the average
        return 0.5 * (dist_trans + dist_rot)

    def lin_interp(self, pose, t):
        """ linearly interpolate the pose values
        :param pose pose to interpoloate to Pose2D
        :param t a number between 0 and 1
        :return the interpolated pose Pose2D"""
        return Pose2D(self.x + t * pose.x, self.y + t * pose.y, self.theta + t * pose.theta)

    def __str__(self):
        return "x : {0:.2f}, y : {1:.2f}, theta : {2}".format(self.x, self.y, self.theta)


class AsteriskTestResults:
    test_type = ["Translation", "Rotation", "Rotation_translation"]
    def __init__(self, name, in_test_type = "none", in_translation_angle="none", in_rotation_angle="none"):
        """distances, angles, index in test
        :param n_samples - number of samples in target paths"""
        self.test_name = name
        self.test_type = in_test_type
        self.translation_angle = in_translation_angle
        self.rotation_angle = in_rotation_angle
        self.end_target_index = -1
        self.dist_target = nan
        self.dist_frechet = nan
        self.dist_along_translation = nan
        self.dist_along_rotation = nan
        self.target_indices = []
        self.obj_poses = []

    def __str__(self):
        """Print results"""
        ret_str = "Test: {0} {1} {2} {3}: ".format(self.test_name, self.test_type, self.translation_angle, self.rotation_angle)
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
        col_names = ["Name", "Type", "TranslationAngle", "RotationAngle"]
        col_names.extend(["DistToTarget", "DistAlongTranslation", "DistAlongRotation", "FrechetDistance"])
        col_names.extend(["LastIndex", "Indices"])
        f.writerow(col_names)

    def write_data(self, f):
        """ Write out data to a csv file
        :param f - csv file writer"""

        row_data = []
        row_data.append(self.test_name)
        row_data.append(self.test_type)
        row_data.append(self.translation_angle)
        row_data.append(self.rotation_angle)
        row_data.append(self.dist_target)
        row_data.append(self.dist_along_translation)
        row_data.append(self.dist_along_rotation)
        row_data.append(self.dist_frechet)
        row_data.append(self.end_target_index)
        for i in self.target_indices:
            row_data.append("{}".format(i))

        f.writerow(row_data)


class AsteriskTestMetrics2D:
    metric_names = {"Distance_along, Distance_target, Frechet_distance"}
    translation_angles = linspace(90, 90-360, 8, endpoint=False)
    rotation_directions = {"Clockwise": -15, "Counterclockwise": 15}
    status_values = {"Successful", "Unsuccessful", "Not_tried"}

    def __init__(self, n_samples=15):
        """
        :param n_samples number of samples in target path"""
        self.target_paths = {}
        self.test_results = []
        self._add_target_paths(n_samples)
        self.reset_test_results()

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
                target_rotation_translation_paths[0][-1].append(Pose2D(x * d, y * d, self.rotation_directions["Clockwise"]))
                target_rotation_translation_paths[1][-1].append(Pose2D(x * d, y * d, self.rotation_directions["Counterclockwise"]))

        target_rotation_paths = [[], []]
        for d in divs:
            target_rotation_paths[0].append(Pose2D(0, 0, d * self.rotation_directions["Clockwise"]))
            target_rotation_paths[1].append(Pose2D(0, 0, d * self.rotation_directions["Counterclockwise"]))

        self.target_paths["Translation"] = target_translation_paths
        self.target_paths["Rotation"] = target_rotation_paths
        self.target_paths["Rotation_translation"] = target_rotation_translation_paths

    @staticmethod
    def _narrow_target(obj_pose, target_poses):
        """ narrown down the closest point on the target poses
        :param obj_pose last object pose Pose2D
        :param target_poses [Pose2D;
        :returns target_i the index of the best match """

        dists_targets = [obj_pose.distance(p) for p in target_poses]
        i_target = dists_targets.index(min(dists_targets))

        return i_target

    @staticmethod
    def _frechet_dist(poses_obj, i_target, target_poses):
        """ Implement Frechet distance
        :param poses_obj all the object poses np.array
        :param i_target the closest point in target_poses
        :param target_poses [Pose2D];
        :returns max of the min distance between target_poses and obj_poses """

        dist_frechet = []
        target_index = []
        n_total = poses_obj.shape[1]
        # Don't search the whole list, just a bracket around where you would expect the closest sample to be
        i_start_search = 0
        step_along = int(floor(1.5 * n_total / i_target)) + 1
        for tp in target_poses[0:i_target]:
            dist_found = 1e30
            i_along_search = i_start_search
            i_end_search = min(i_start_search + step_along, n_total)
            for i_p in range(i_start_search, i_end_search):
                p = poses_obj[:, i_p]
                dist = tp.distance(Pose2D(p[0], p[1], p[2]))
                if dist < dist_found:
                    dist_found = dist
                    i_along_search = i_p

            print("{} {} {}".format(i_start_search, i_along_search, dist_found))
            target_index.append(i_along_search)
            i_start_search = i_along_search + 1
            dist_frechet.append(dist_found)

        return max(dist_frechet), target_index

    def add_translation_test(self, name, in_which, poses_obj):
        """Add the translation test
        :param name: str - name of test, eg, hand type
        :param in_which is 0..7, which angle in the asterisk
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        test_type = AsteriskTestResults.test_type[0]
        target_poses = self.target_paths[test_type][in_which]
        target_pose = target_poses[-1]
        ret_dists = AsteriskTestResults(name, test_type, in_translation_angle=dir_options_no_rot[in_which])
        ret_dists.obj_poses = poses_obj.copy()
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        n_total = poses_obj.shape[1]

        # Check that we're in at least roughly the right ballpark for the end pose
        last_pose_angle = 180.0 * arctan2(last_pose_obj.y, last_pose_obj.x) / pi
        expected_angle = self.translation_angles[in_which]
        if last_pose_angle - expected_angle > 180:
            last_pose_angle -= 360
        elif expected_angle - last_pose_angle > 180:
            last_pose_angle += 360

        if abs(last_pose_angle - expected_angle) > 65:
            print("Warning: Translation {} detected bad last pose {}, expected {}".format(in_which, last_angle,
                                                                                          self.translation_angles[
                                                                                              in_which]))

        ret_dists.dist_along_translation = sqrt(max([poses_obj[0, i_p]**2 + poses_obj[1, i_p]**2 for i_p in range(0, n_total)]))
        ret_dists.end_target_index = self._narrow_target(last_pose_obj, target_poses)

        ret_dists.dist_target = last_pose_obj.distance(target_pose)

        if ret_dists.end_target_index == 0:
            print("Warning: Closest pose was first pose")
            ret_dists.end_target_index += 1

        ret_dists.dist_frechet, ret_dists.target_indices = self._frechet_dist(poses_obj, ret_dists.end_target_index, target_poses)

        self.test_results.append(ret_dists)
        return ret_dists

    def add_rotation_test(self, name, in_which, poses_obj):
        """Add the translation test
        :param in_which is Clockwise or Counterclockwise
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        test_type = AsteriskTestResults.test_type[1]
        target_poses = self.target_paths[test_type][in_which]
        target_pose = target_poses[-1]
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        ret_dists = AsteriskTestResults(name, test_type, in_rotation_angle=type_options[in_which+1])
        ret_dists.obj_poses = poses_obj.copy()

        # Check that we're in at least roughly the right ballpark for the end pose
        dist_from_center = sqrt(last_pose_obj.x**2 + last_pose_obj.y**2)
        if dist_from_center > 0.2:
            print("Warning: rotation test {} had big offset {}".format(in_which, dist_from_center))

        ret_dists.dist_along_rotation = abs(last_pose_obj.theta) / self.rotation_directions["Counterclockwise"]
        ret_dists.end_target_index = self._narrow_target(last_pose_obj, target_poses)

        ret_dists.dist_target = last_pose_obj.distance(target_pose)

        if ret_dists.end_target_index == 0:
            print("Warning: Closest pose was first pose")
            ret_dists.end_target_index += 1

        ret_dists.dist_frechet, ret_dists.target_indices = self._frechet_dist(poses_obj, ret_dists.end_target_index, target_poses)

        self.test_results.append(ret_dists)
        return ret_dists

    def add_rotation_translation_test(self, name, in_which_rot, in_which_trans, poses_obj):
        """Add the translation test
        :param in_which_rot is 0 or 1, clockwise or counter
        :param in_which_trans is 0..7, which angle in the asterisk
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        test_type = AsteriskTestResults.test_type[2]
        target_poses = self.target_paths[test_type][in_which_rot][in_which_trans]
        target_pose = target_poses[-1]
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        n_total = poses_obj.shape[1]
        ret_dists = AsteriskTestResults(name, test_type,
                                        in_translation_angle=dir_options_no_rot[in_which_trans],
                                        in_rotation_angle=type_options[in_which_rot+1])
        ret_dists.obj_poses = poses_obj.copy()

        # Check that we're in at least roughly the right ballpark for the end pose
        last_pose_angle = 180.0 * arctan2(last_pose_obj.y, last_pose_obj.x) / pi
        expected_angle = self.translation_angles[in_which_trans]
        if last_pose_angle - expected_angle > 180:
            last_pose_angle -= 360
        elif expected_angle - last_pose_angle > 180:
            last_pose_angle += 360

        if abs(last_pose_angle - expected_angle) > 65:
            print("Warning: Translation {} detected bad last pose {}, expected {}".format(in_which_trans, last_angle,
                                                                                          self.translation_angles[
                                                                                              in_which_trans]))

        ret_dists.dist_along_translation = sqrt(max([poses_obj[0, i_p]**2 + poses_obj[1, i_p]**2 for i_p in range(0, n_total)]))
        ret_dists.dist_along_rotation = abs(last_pose_obj.theta) / self.rotation_directions["Counterclockwise"]

        ret_dists.end_target_index = self._narrow_target(last_pose_obj, target_poses)

        ret_dists.dist_target = last_pose_obj.distance(target_pose)

        if ret_dists.end_target_index == 0:
            print("Warning: Closest pose was first pose")
            ret_dists.end_target_index += 1

        ret_dists.dist_frechet, ret_dists.target_indices = self._frechet_dist(poses_obj, ret_dists.end_target_index, target_poses)
        self.test_results.append(ret_dists)
        return ret_dists

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

        for i_ang, ang in enumerate(linspace(0, self.rotation_directions["Clockwise"], n_poses)):
            obj_poses[0, i_ang] = noise_x[i_ang]
            obj_poses[1, i_ang] = noise_y[i_ang]
            obj_poses[2, i_ang] = ang + noise_ang[i_ang]

        dists = self.add_rotation_test("Test", 0, obj_poses)
        print("Rotation, angle {0}, distances {1}".format("Clockwise", dists))
        for i_ang, ang in enumerate(linspace(0, self.rotation_directions["Counterclockwise"], n_poses)):
            obj_poses[2, i_ang] = ang + noise_ang[i_ang]
        dists = self.add_rotation_test("Test", 1, obj_poses)
        print("Rotation, angle {0}: {1}".format("Counterclockwise", dists))

    def test_rotation_translation(self):
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
                obj_poses[2, i_p] = self.rotation_directions["Clockwise"] + noise_ang[i_ang, i_p]

            dists = self.add_rotation_translation_test("Test", 0, i_ang, obj_poses)
            print("Rotation {} Translation, angle {}: {}".format("Clockwise", ang, dists))

        for i_ang, ang in enumerate(self.translation_angles):
            for i_p, div in enumerate(linspace(0, 1, n_poses)):
                obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
                obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
                obj_poses[2, i_p] = self.rotation_directions["Counterclockwise"] + noise_ang[i_ang, i_p]

            dists = self.add_rotation_translation_test("Test", 1, i_ang, obj_poses)
            print("Rotation {} Translation, angle {}: {}".format("Counterclockwise", ang, dists))

    @staticmethod
    def run_tests():
        my_asterisk_tests = AsteriskTestMetrics2D()

        my_asterisk_tests.test_translation()
        my_asterisk_tests.test_rotation()
        my_asterisk_tests.test_rotation_translation()

        my_asterisk_tests.write_test_results("test_results.csv")

    def process_file(self, name: str, trial_type: str, ang_name: str, obj_poses):
        """Read the files, compute the metrics
        :param name subject + hand + trial name
        :param trial_type from asterisk_0_prompts, minus, plus, etc
        :param ang_name from asterisk_0_prompts, none or a, b, c etc
        :param obj_poses object poses
        :returns Distances"""

        print("{0}\n x {1} y {2} t {3}".format(trial_type+ang_name, obj_poses[0,-1], obj_poses[1, -1], obj_poses[2, -1]))
        ang = ord(ang_name[0]) - ord('a')
        if trial_type == "minus15":
            dists = self.add_rotation_translation_test(name, 1, ang, obj_poses)
        elif trial_type == "plus15":
            dists = self.add_rotation_translation_test(name, 0, ang, obj_poses)
        elif ang_name == "cw":
            dists = self.add_rotation_test(name, 1, obj_poses)
        elif ang_name == "ccw":
            dists = self.add_rotation_test(name, 0, obj_poses)
        else:
            dists = self.add_translation_test(name, ang, obj_poses)

        return dists

    @staticmethod
    def process_files(dir_name, subject_name, hand):
        """Read the files, compute the metrics
        !param dir_name input file name
        :param subject_name name of subject to process
        :param hand name of hand to process
        :return my_tests Array of AsteriskTestMetrics with tests"""

        my_tests = [AsteriskTestMetrics2D() for _ in range(0, 5)]
        for fname in generate_fname(dir_name, subject_name, hand):
            fname_pieces = fname.split("_")
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
                    print("{0}\n x {1} y {2} t {3}".format(fname, obj_poses[0,-1], obj_poses[1, -1], obj_poses[2, -1]))
                    trial_number = int(fname_pieces[-1][0])-1
                    trial_type = fname_pieces[-2]
                    angle_name = fname_pieces[-3]
                    name = subject_name + "_" + fname_pieces[-4] + "_" + "Trial{0}".format(trial_number)
                    dists = my_tests[trial_number].process_file(name, trial_type, angle_name, obj_poses)

                    print("{0}".format(dists))
            except FileNotFoundError:
                print("File not found: {0}".format(fname))

        return my_tests

if __name__ == '__main__':
    #AsteriskTestMetrics2D.run_tests()

    #dir_name_process = "/Users/grimmc/Box/Grasping/asterisk_test_data/filtered_data/"
    dir_name_process = "/Users/grimmc/Downloads/filtered/"
    subject_name_process = "filt_josh"
    hand_process = "2v3"
    my_test_results = AsteriskTestMetrics2D.process_files(dir_name_process, subject_name_process, hand_process)

    for i, t in enumerate(my_test_results):
        t.write_test_results("check_res{0}.csv".format(i))
