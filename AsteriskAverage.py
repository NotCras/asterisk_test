#!/usr/bin/env python3

from AsteriskTestMetrics import AsteriskTestResults, Pose2D, AsteriskTestMetrics2D
from math import sqrt

class AsteriskAverage(AsteriskTestResults):
    def __init__(self, name):
        super().__init__(name)
        self.pose_average = []
        self.pose_sd = []

    def set_test_type_data(self, atr: AsteriskTestResults):
        """ Set the name etc of this test
        atr - an AsteriskTestResult"""
        self.test_name = atr.test_name
        self.test_type = atr.test_type
        self.translation_angle = atr.translation_angle
        self.rotation_angle = atr.rotation_angle

    def set_average(self, obj_poses: [[Pose2D]], atrs: [AsteriskTestResults]):
        """ Average 2 or more trials for one test
        :param obj_poses - array of array of poses one for each trial
        :param atrs = array of AsteriskTestResults
        :returns array of average poses with += poses
        """
        self.pose_average = []
        self.pose_sd = []
        try:
            self.set_test_type_data(atrs[0])

            # This is really clunky, but it's the easiest way to deal
            # with the problem that the arrays have different sizes...
            n_max = max([len(t.target_indices) for t in atrs])
            self.pose_average = [Pose2D()] * n_max
            sd_dist = [0] * n_max
            sd_theta = [0] * n_max
            count = [0] * n_max
            for i, t in enumerate(atrs):
                for j, index in enumerate(t.target_indices):
                    self.pose_average[j].x += obj_poses[i][0, index]
                    self.pose_average[j].y += obj_poses[i][1, index]
                    self.pose_average[j].theta += obj_poses[i][2, index]
                    count[j] += 1

            # Average
            for i, p in enumerate(self.pose_average):
                self.pose_average[i].x /= count[i]
                self.pose_average[i].y /= count[i]
                self.pose_average[i].theta /= count[i]
                count[i] = 0

            # SD - do theta separately from distance to centerline
            for i, t in enumerate(atrs):
                for j, index in enumerate(t.target_indices):
                    dx = self.pose_average[j].x - obj_poses[i][0, index]
                    dy = self.pose_average[j].y - obj_poses[i][1, index]
                    dist = sqrt(dx * dx + dy * dy)
                    dt = abs(self.pose_average[j].theta - obj_poses[i][2, index])
                    sd_theta[i] += dt
                    sd_dist[i] += dist
                    count[j] += 1

            # Normalize SD
            last_valid_i = 0
            for i, p in enumerate(self.pose_average):
                if count[i] > 1:
                    self.pose_sd.append((sd_dist[i] / (count[i] - 1), sd_theta[i] / (count[i] - 1)))
                    last_valid_i = i
                else:
                    self.pose_sd.append( self.pose_sd[i] )

        except IndexError:
            pass


if __name__ == '__main__':
    dir_name_process = "/Users/grimmc/Downloads/filtered/"
    subject_name_process = "filt_josh"
    hand_process = "2v3"
    my_test_results = AsteriskTestMetrics2D.process_files(dir_name_process, subject_name_process, hand_process)

    n_tests = len(my_test_results[0].test_results)

    at_avgs = []
    for it in range(0, n_tests):
        obj_poses = []
        ret_tests = []
        at_avg = AsteriskAverage(my_test_results[0].test_results[it].test_name)
        for i, t in enumerate(my_test_results):
            try:
                obj_poses.append(my_test_results[i].test_results[it].obj_poses)
                ret_tests.append(my_test_results[i].test_results[it])
            except IndexError:
                pass
        at_avg.set_average(obj_poses, ret_tests)
        at_avgs.append(at_avg)

