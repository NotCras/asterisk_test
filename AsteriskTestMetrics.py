#!/usr/bin/env python3
from typing import List, Any, Tuple

import numpy as np
from numpy import sin, cos, pi, linspace, sqrt, abs, arctan2, zeros, floor, nan
import csv
from asterisk_0_prompts import generate_fname


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
    def __init__(self, name):
        """distances, angles, index in test
        :param n_samples - number of samples in target paths"""
        self.test_name = name
        self.end_target_index = -1
        self.dist_target = nan
        self.dist_frechet = nan
        self.dist_along = nan
        self.target_indices = []

    def __str__(self):
        """Print results"""
        if self.end_target_index == -1:
            return "Test: {0}, no result".format(self.test_name)
        if len(self.dist_along) == 2:
            return "Test: {0} Along: d {1:0.3f} t{2:0.3f} Target: {3:0.3f} Frechet: {4:0.3f}\n ".format(self.test_name,
                                                                                                        self.dist_along[0],
                                                                                                        self.dist_along[1],
                                                                                                        self.dist_target,
                                                                                                        self.dist_frechet)

        return "Test: {0} Along: {1:0.3f} Target: {2:0.3f} Frechet: {3:0.3f}\n ".format(self.test_name,
                                                                                        self.dist_along,
                                                                                        self.dist_target,
                                                                                        self.dist_frechet)


class AsteriskTestMetrics2D:
    def __init__(self, n_samples=15):
        """
        :param n_samples number of samples in target path"""

        self.test_names = {"Translation": 0, "Rotation": 1, "Rotation_translation": 2}
        self.metric_names = {"Distance_along, Distance_target, Frechet_distance"}
        self.translation_angles = linspace(90, 90-360, 8, endpoint=False)
        self.rotation_directions = {"Clockwise": -15, "Counterclockwise": 15}
        self.status_values = {"Successful", "Unsuccessful", "Not_tried"}

        self.target_paths = {}
        self.test_results = {}
        self._add_target_paths(n_samples)
        self.reset_test_results()
        self.object_paths = {}

    def reset_test_results(self):
        """Zero out the results"""

        for t in self.test_names:
            self.test_results[t] = []

        self.test_results["Rotation"].append(AsteriskTestResults("Rotation_cw", n_samples))
        self.test_results["Rotation"].append(AsteriskTestResults("Rotation_ccw", n_samples))

        self.test_results["Rotation_translation"].append([])
        self.test_results["Rotation_translation"].append([])
        for a in self.translation_angles:
            self.test_results["Translation"].append(AsteriskTestResults("Translation {0}".format(a), n_samples))
            self.test_results["Rotation_translation"][0].append(AsteriskTestResults("Rotation_translation_cw {0}".format(a), n_samples))
            self.test_results["Rotation_translation"][1].append(AsteriskTestResults("Rotation_translation_ccw {0}".format(a), n_samples))

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

    def add_translation_test(self, in_which, poses_obj):
        """Add the translation test
        :param in_which is 0..7, which angle in the asterisk
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        target_poses = self.target_paths["Translation"][in_which]
        target_pose = target_poses[-1]
        res = self.test_results["Translation"][in_which]
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        n_total = poses_obj.shape[1]

        # Check that we're in at least roughly the right ballpark for the end pose
        last_angle = 180.0 * arctan2(last_pose_obj.y, last_pose_obj.x) / pi
        if last_angle < 0:
            last_angle += 360

        if in_which == 0 and last_angle > 180:
            last_angle -= 360
        elif in_which == 7 and last_angle < 180:
            last_angle += 360

        if abs(last_angle - self.translation_angles[in_which]) > 65:
            print("Warning: Translation {} detected bad last pose {}, expected {}".format(in_which, last_angle,
                                                                                          self.translation_angles[
                                                                                              in_which]))

        res.dist_along = sqrt(max([poses_obj[0, i_p]**2 + poses_obj[1, i_p]**2 for i_p in range(0, n_total)]))
        i_target = self._narrow_target(last_pose_obj, target_poses)

        res.dist_target = last_pose_obj.distance(target_pose)

        if i_target == 0:
            print("Warning: Closest pose was first pose")
            i_target += 1

        res.dist_frechet, res.target_index = self._frechet_dist(poses_obj, i_target, target_poses)

        return res

    def add_rotation_test(self, in_which, poses_obj):
        """Add the translation test
        :param in_which is Clockwise or Counterclockwise
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        target_poses = self.target_paths["Rotation"][in_which]
        target_pose = target_poses[-1]
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])

        # Check that we're in at least roughly the right ballpark for the end pose
        dist_from_center = sqrt(last_pose_obj.x**2 + last_pose_obj.y**2)
        if dist_from_center > 0.2:
            print("Warning: rotation test {} had big offset {}".format(in_which, dist_from_center))

        dist_along = abs(last_pose_obj.theta) / self.rotation_directions["Counterclockwise"]
        i_target = self._narrow_target(last_pose_obj, target_poses)

        dist_target = last_pose_obj.distance(target_pose)

        if i_target == 0:
            print("Warning: Closest pose was first pose")
            i_target += 1

        dist_frechet = self._frechet_dist(poses_obj, i_target, target_poses)

        self.test_results["Rotation"][in_which] = (dist_along, dist_target, dist_frechet)

        return dist_along, dist_target, dist_frechet

    def add_rotation_translation_test(self, in_which_rot, in_which_trans, poses_obj):
        """Add the translation test
        :param in_which_rot is 0 or 1, clockwise or counter
        :param in_which_trans is 0..7, which angle in the asterisk
        :param poses_obj is a 3xn matrix of x,y,theta poses
        :returns Percentage distance traveled, percentage error last pose, overall path score"""

        target_poses = self.target_paths["Rotation_translation"][in_which_rot][in_which_trans]
        target_pose = target_poses[-1]
        last_pose_obj = Pose2D(poses_obj[0, -1], poses_obj[1, -1], poses_obj[2, -1])
        n_total = poses_obj.shape[1]

        # Check that we're in at least roughly the right ballpark for the end pose
        last_angle = 180.0 * arctan2(last_pose_obj.y, last_pose_obj.x) / pi
        if last_angle < 0:
            last_angle += 360

        if in_which_trans == 0 and last_angle > 180:
            last_angle -= 360
        elif in_which_trans == 7 and last_angle < 180:
            last_angle += 360

        if abs(last_angle - self.translation_angles[in_which_trans]) > 65:
            print("Warning: Translation {} detected bad last pose {}, expected {}".format(in_which_trans, last_angle,
                                                                                          self.translation_angles[
                                                                                              in_which_trans]))

        dist_along_trans = sqrt(max([poses_obj[0, i_p]**2 + poses_obj[1, i_p]**2 for i_p in range(0, n_total)]))
        dist_along_rot = abs(last_pose_obj.theta) / self.rotation_directions["Counterclockwise"]

        i_target = self._narrow_target(last_pose_obj, target_poses)

        dist_target = last_pose_obj.distance(target_pose)

        if i_target == 0:
            print("Warning: Closest pose was first pose")
            i_target += 1

        dist_frechet = self._frechet_dist(poses_obj, i_target, target_poses)

        self.test_results["Rotation_translation"][in_which_rot][in_which_trans] = ((dist_along_trans, dist_along_rot), dist_target, dist_frechet)

        return (dist_along_trans, dist_along_rot), dist_target, dist_frechet


def test_translation(my_tests):
    """ Make fake translation data and add it
    :param my_tests AsteriskTestMetrics2D"""
    from numpy.random import uniform

    n_poses = 100
    obj_poses = zeros([3, n_poses])
    noise_x = uniform(-0.1, 0.1, (8, n_poses))
    noise_y = uniform(-0.1, 0.1, (8, n_poses))
    noise_ang = uniform(-5, 5, (8, n_poses))

    for i_ang, ang in enumerate(my_tests.translation_angles):
        for i_p, div in enumerate(linspace(0, 1, n_poses)):
            obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
            obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
            obj_poses[2, i_p] = noise_ang[i_ang, i_p]

        dists = my_tests.add_translation_test(i_ang, obj_poses)
        print("Translation, angle {0}, distances {1}".format(ang, dists))


def test_rotation(my_tests):
    """ Make fake translation data and add it
    :rtype: None
    :param my_tests AsteriskTestMetrics2D"""
    from numpy.random import uniform

    n_poses = 50
    obj_poses = zeros([3, n_poses])
    noise_x = uniform(-0.1, 0.1, n_poses)
    noise_y = uniform(-0.1, 0.1, n_poses)
    noise_ang = uniform(-0.5, 0.5, n_poses)

    for i_ang, ang in enumerate(linspace(0, my_tests.rotation_directions["Clockwise"], n_poses)):
        obj_poses[0, i_ang] = noise_x[i_ang]
        obj_poses[1, i_ang] = noise_y[i_ang]
        obj_poses[2, i_ang] = ang + noise_ang[i_ang]

    dists = my_tests.add_rotation_test(0, obj_poses)
    print("Rotation, angle {0}, distances {1}".format("Clockwise", dists))
    for i_ang, ang in enumerate(linspace(0, my_tests.rotation_directions["Counterclockwise"], n_poses)):
        obj_poses[2, i_ang] = ang + noise_ang[i_ang]
    dists = my_tests.add_rotation_test(1, obj_poses)
    print("Rotation, angle {0}, distances {1}".format("Counterclockwise", dists))


def test_rotation_translation(my_tests):
    """ Make fake translation data and add it
    :param my_tests AsteriskTestMetrics2D"""
    from numpy.random import uniform

    n_poses = 100
    obj_poses = zeros([3, n_poses])
    noise_x = uniform(-0.1, 0.1, (8, n_poses))
    noise_y = uniform(-0.1, 0.1, (8, n_poses))
    noise_ang = uniform(-5, 5, (8, n_poses))

    for i_ang, ang in enumerate(my_tests.translation_angles):
        for i_p, div in enumerate(linspace(0, 1, n_poses)):
            obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
            obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
            obj_poses[2, i_p] = my_tests.rotation_directions["Clockwise"] + noise_ang[i_ang, i_p]

        dists = my_tests.add_rotation_translation_test(0, i_ang, obj_poses)
        print("Rotation {} Translation, angle {}, distances {}".format("Clockwise", ang, dists))

    for i_ang, ang in enumerate(my_tests.translation_angles):
        for i_p, div in enumerate(linspace(0, 1, n_poses)):
            obj_poses[0, i_p] = div * cos(pi*ang/180) + noise_x[i_ang, i_p]
            obj_poses[1, i_p] = div * sin(pi*ang/180) + noise_y[i_ang, i_p]
            obj_poses[2, i_p] = my_tests.rotation_directions["Counterclockwise"] + noise_ang[i_ang, i_p]

        dists = my_tests.add_rotation_translation_test(1, i_ang, obj_poses)
        print("Rotation {} Translation, angle {}, distances {}".format("Counterclockwise", ang, dists))


def run_tests():
    my_asterisk_tests = AsteriskTestMetrics2D()

    test_translation(my_asterisk_tests)
    test_rotation(my_asterisk_tests)
    test_rotation_translation(my_asterisk_tests)


def process_files(dir_name, subject_name, hand, my_tests):
    """Read the files, compute the metrics
    !param dir_name input file name
    :param subject_name name of subject to process
    :param hand name of hand to process
    :param my_tests Array of AsteriskTestMetrics to put tests in"""

    ret_dists: List[Tuple[Any, Any]] = []
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
                ang = ord(fname_pieces[-3][0]) - ord('a')
                if trial_type is "minus15":
                    dists = my_tests[trial_number].add_rotation_translation_test(1, ang, obj_poses)
                elif trial_type is "plus15":
                    dists = my_tests[trial_number].add_rotation_translation_test(0, ang, obj_poses)
                elif fname_pieces[-3] is "cw":
                    dists = my_tests[trial_number].add_rotation_test(0, obj_poses)
                elif fname_pieces[-3] is "ccw":
                    dists = my_tests[trial_number].add_rotation_test(1, obj_poses)
                else:
                    dists = my_tests[trial_number].add_translation_test(ang, obj_poses)
    
                ret_dists.append((fname, dists))
                print("{0} dists {1} ".format(fname, dists))
        except FileNotFoundError:
            print("File not found: {0}".format(fname))

    return ret_dists


if __name__ == '__main__':
    my_tests = [AsteriskTestMetrics2D() for i in range(0, 3)]

    dir_name_process = "/Users/grimmc/Box/Grasping/asterisk_test_data/filtered_data/"
    subject_name_process = "filt_josh"
    hand_process = "2v2"
    ret_dists = process_files(dir_name_process, subject_name_process, hand_process, my_tests)
