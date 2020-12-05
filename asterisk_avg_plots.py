#!/usr/bin/env python3

from math import pi, cos, sin, sqrt
import matplotlib.pyplot as plt

from AsteriskTestMetrics import Pose2D, AsteriskTestMetrics2D, AsteriskTestResults
from AsteriskAverage import AsteriskAverage


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


def plot_frechet_translation(my_asterisk: AsteriskTestMetrics2D) -> None:
    """ Plot the path and the frechet match
    :param - my_asterisk - the test metrics"""

    # Find the results that are translations
    trans_res = my_asterisk.get_test_results(AsteriskTestResults.test_type[0])

    for res in trans_res:
        plt.plot(res.obj_poses[0, :], res.obj_poses[1, :], color="red", linestyle='dotted')
        for i, index in enumerate(res.target_indices):
            plt.plot([res.target_path[i].x, res.obj_poses[0, index]],
                     [res.target_path[i].y, res.obj_poses[1, index]],
                     linestyle='-', color="black", marker='.')
            plot_pose(Pose2D(res.obj_poses[0, index], res.obj_poses[1, index], res.obj_poses[2, index]), scl=0.5)


def plot_average_translation(asterisk_avg: AsteriskAverage):
    """Plot the average path
    :param asterisk_avg - the computed averages"""

    # Offset vectors
    vec_offset = []
    for i in range(0, len(asterisk_avg.pose_average)):
        try:
            prev_x = asterisk_avg.pose_average[i-1].x
            prev_y = asterisk_avg.pose_average[i-1].y
        except IndexError:
            prev_x = asterisk_avg.pose_average[i].x
            prev_y = asterisk_avg.pose_average[i].y
        try:
            next_x = asterisk_avg.pose_average[i+1].x
            next_y = asterisk_avg.pose_average[i+1].y
        except IndexError:
            next_x = asterisk_avg.pose_average[i].x
            next_y = asterisk_avg.pose_average[i].y
        dx = next_x - prev_x
        dy = next_y - prev_y
        dlen = sqrt(dx * dx + dy * dy)
        vec_offset.append((asterisk_avg.pose_sd * -dy / dlen, asterisk_avg.pose_sd * dx / dlen))

    poly = []
    for a, v in zip(asterisk_avg.pose_average, vec_offset):
        poly.append([a.x + vec_offset[0], a.y + vec_offset[1]])

    for a, v in zip(reversed(asterisk_avg.pose_average), reversed(vec_offset)):
        poly.append([a.x - vec_offset[0], a.y - vec_offset[1]])

    polyg = plt.Polygon(poly, color='green', alpha=0.5)
    plt.gca().add_patch(polyg)

    # Line down the middle
    plt.plot([p.x for p in asterisk_avg.pose_average],
             [p.y for p in asterisk_avg.pose_average],
             linestyle='-', color="black")
    # Poses at dots
    for p in zip(asterisk_avg.pose_average):
        plot_pose(Pose2D(p[0], p[1], p[2]))


if __name__ == '__main__':
    dir_name_process = "/Users/grimmc/Downloads/filtered/"
    subject_name_process = "filt_josh"
    hand_process = "2v3"
    my_test_results = AsteriskTestMetrics2D.process_files(dir_name_process, subject_name_process, hand_process)

    fig, axs = plt.subplots(2, len(my_test_results))

    for i, tr in enumerate(my_test_results):
        plt.axes(axs[0, i])
        axs[0, i].set_aspect('equal', adjustable='box')
        plot_star(tr.target_paths)
        plot_frechet_translation(tr)

    # Find the results that are translations
    trans_res = []
    for it in my_test_results:
        trans_res.extend(it.get_test_results(AsteriskTestResults.test_type[0]))

    avg_tests = AsteriskAverage(trans_res[0].test_name)
    avg_tests.average(trans_res)

    plt.axes(axs[1, 0])
    axs[1, 0].set_aspect('equal', adjustable='box')
    plot_star(my_test_results[0].target_paths)
    plot_average_translation(avg_tests)

    plt.show(block=True)


