#!/usr/bin/env python3

from AsteriskTestMetrics import AsteriskTestResults, Pose2D, AsteriskTestMetrics2D
from asterisk_0_prompts import AsteriskTestTypes
from math import sqrt

class AsteriskAverage(AsteriskTestTypes):
    def __init__(self, ):
        super().__init__()

        self.names = []
        self.pose_average = []
        self.pose_sd = []

    def set_average(self, atrs: [AsteriskTestResults]):
        """ Average 2 or more trials for one test
        :param atrs = array of AsteriskTestResults
        :returns array of average poses with += poses
        """

        # initializing
        self.pose_average = []
        self.pose_sd = []
        try:
            # Sets type
            self.set(atrs[0])
        except IndexError:
            pass

        # This is really clunky, but it's the easiest way to deal
        # with the problem that the arrays have different sizes...
        n_max = max([len(t.target_indices) for t in atrs]) # get max length of all data arrays for one trial
        self.pose_average = [Pose2D() for _ in range(0, n_max)]
        sd_dist = [0] * n_max
        sd_theta = [0] * n_max
        count = [0] * n_max
        for t in atrs:
            self.names.append(t.test_name)

            for j, index in enumerate(t.target_indices):
                print("{0} {1} {2}".format(index, t.obj_poses[0, index], t.obj_poses[1, index]))
                self.pose_average[j].x += t.obj_poses[0, index]
                self.pose_average[j].y += t.obj_poses[1, index]
                self.pose_average[j].theta += t.obj_poses[2, index]
                count[j] += 1

        # Average
        for i, c in enumerate(count):
            self.pose_average[i].x /= c
            self.pose_average[i].y /= c
            self.pose_average[i].theta /= c
            count[i] = 0

        # SD - do theta separately from distance to centerline
        for t in atrs:
            for j, index in enumerate(t.target_indices):
                p = t.obj_poses[:, index]
                dx = self.pose_average[j].x - p[0]
                dy = self.pose_average[j].y - p[1]
                dist = sqrt(dx * dx + dy * dy)
                dt = abs(self.pose_average[j].theta - p[2])
                sd_theta[j] += dt
                sd_dist[j] += dist
                count[j] += 1

        # Normalize SD
        last_valid_i = 0
        for i, p in enumerate(self.pose_average):
            if count[i] > 1:
                self.pose_sd.append((sd_dist[i] / (count[i] - 1), sd_theta[i] / (count[i] - 1)))
                last_valid_i = i
            else:
                self.pose_sd.append(self.pose_sd[last_valid_i])


if __name__ == '__main__':
    dir_name_process = "/Users/grimmc/Downloads/filtered/"
    subject_name_process = "filt_josh"
    hand_process = "2v3"
    my_test_results = AsteriskTestMetrics2D.process_files(dir_name_process, subject_name_process, hand_process)

    at_avgs = []
    for tt in AsteriskTestTypes.generate_translation():
        ret_tests = []
        for trial_t in my_test_results:
            ret_tests.extend(trial_t.get_test_results(tt))

        at_avg = AsteriskAverage()
        at_avg.set_average(ret_tests)

        at_avgs.append(at_avg)

