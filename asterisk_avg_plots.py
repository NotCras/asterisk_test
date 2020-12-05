#!/usr/bin/env python3

import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt

from AsteriskTestMetrics import Pose2D, AsteriskTestMetrics2D, AsteriskTestResults


def plot_pose(pose: Pose2D, rgb=(0.5, 0.5, 0.5), scl=1.0) -> None:
    """ Plot a single pose as a circle with a line for orientation
    :param pose - the pose,
    :param rgb - the color, default grey,
    :param scl - amount to scale marker by"""

    ms = str(max(int(10.0 * scl), 1))
    plt.plot(pose.x, pose.y, marker='o', markersize=ms, color=rgb)
    dx = scl * 0.02 * cos(pi * pose.theta/180.0)
    dy = scl * 0.02 * sin(pi * pose.theta/180.0)
    plt.plot([pose.x, pose.x+dx], [pose.y, pose.y+dy], linewidth=0.5, color=(0, 0, 0))


def plot_star(target_paths: [Pose2D]):
    """ Plot the star in the middle
     :param - target_paths - all of the target paths from AsteriskTestMetrics2D"""

    for path in target_paths["Translation"]:
        plt.plot([0, path[-1].x], [0, path[-1].y], color="grey", linestyle='--')
        plot_pose(path[-1])


def plot_frechet_translation(my_asterisk: AsteriskTestMetrics2D):
    """ Plot the path and the frechet match
    :param atr - the asterisk test result
    :param - my_asterisk - the test metrics
    :param obj_poses - the object path"""

    # Find the results that are translations
    trans_res = my_asterisk.get_test_results(AsteriskTestResults.test_type[0])

    for res in trans_res:
        plt.plot(res.obj_poses[0, :], res.obj_poses[1, :], color="red", linestyle='dotted')
        for i, index in enumerate(res.target_indices):
            plt.plot([res.target_path[i].x, res.obj_poses[0, index]],
                     [res.target_path[i].y, res.obj_poses[1, index]],
                     linestyle='-', color="black", marker='.')
            plot_pose(Pose2D(res.obj_poses[0, index], res.obj_poses[1, index], res.obj_poses[2, index]), scl=0.5)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    dir_name_process = "/Users/grimmc/Downloads/filtered/"
    subject_name_process = "filt_josh"
    hand_process = "2v3"
    my_test_results = AsteriskTestMetrics2D.process_files(dir_name_process, subject_name_process, hand_process)

    n_tests = len(my_test_results[0].test_results)

    for it in range(0, n_tests):
        fig.clf()
        ax.set_aspect('equal', adjustable='box')
        plot_star(my_test_results[it].target_paths)
        plot_frechet_translation(my_test_results[it])

    plt.show(block=True)



