import math as m
import numpy as np
import pdb

from matplotlib import pyplot as plt
import data_manager as datamanager


class AsteriskPlotting:
    """
    Resources for asterisk plots
    """
    colors = ["tab:blue", "tab:purple", "tab:red", "tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

    def __init__(self):
        pass

    @staticmethod
    def get_dir_color(dir):  # TODO: finalize colors based on color wheel?
        colors = {"a": "tab:blue", "b": "tab:purple", "c": "tab:red", "d": "tab:olive",
                  "e": "tab:cyan", "f": "tab:green", "g": "tab:pink", "h": "tab:orange",
                  "no": "tab:blue", "ne": "tab:purple", "ea": "tab:red", "se": "tab:olive",
                  "so": "tab:cyan", "sw": "tab:green", "we": "tab:pink", "nw": "tab:orange"
                  }
        return colors[dir]

    @staticmethod
    def round_half_up(n, decimals=0):
        """
        Used for plotting
        # from: https://realpython.com/python-rounding/
        """
        multiplier = 10 ** decimals
        return m.floor(n*multiplier + 0.5) / multiplier

    @staticmethod
    def round_half_down(n, decimals=0):
        """
        Used for plotting
        # from: https://realpython.com/python-rounding/
        """
        multiplier = 10 ** decimals
        return m.ceil(n*multiplier - 0.5) / multiplier

    # well, what I can do is do a linspace for both x and y...
    # its straightforward because these are perfect lines we are drawing
    @staticmethod
    def straight(num_points=25, mod=1, max=0.5):
        vals = np.linspace(0, max, num_points)
        z = np.zeros(num_points)

        set1 = mod * vals
        return set1, z

    @staticmethod
    def diagonal(num_points=25, mod1=1, mod2=1, max=0.3536):
        coords = np.linspace(0, max, num_points)  # 0.3536 ~= 1 / (2* sqrt(2))

        set1 = mod1 * coords
        set2 = mod2 * coords

        return set1, set2

    @staticmethod
    def get_a(num_points=25, max=0.5):
        y_coords, x_coords = AsteriskPlotting.straight(num_points=num_points, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_b(num_points=25, max=0.3536):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points=num_points, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_c(num_points=25, max=0.5):
        x_coords, y_coords = AsteriskPlotting.straight(num_points=num_points, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_d(num_points=25, max=0.3536):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points=num_points, mod1=1, mod2=-1, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_e(num_points=25, max=0.5):
        y_coords, x_coords = AsteriskPlotting.straight(num_points, mod=-1, max=max) #, max=0.75)
        return x_coords, y_coords

    @staticmethod
    def get_f(num_points=25, max=0.3536):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points=num_points, mod1=-1, mod2=-1, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_g(num_points=25, max=0.5):
        x_coords, y_coords = AsteriskPlotting.straight(num_points, mod=-1, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_h(num_points=25, max=0.3536):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points, mod1=-1, max=max)
        return x_coords, y_coords

    @staticmethod
    def get_direction(translation_label, n_samples=100):
        """ # TODO: adapt for new t labels
        return the appropriate x and y coordinates for any direction
        """
        # TODO: make it so we can customize the amount in each direction
        x_vals, y_vals = 0, 0
        if translation_label in ["a", "no"]:
            x_vals, y_vals = AsteriskPlotting.get_a(n_samples)
        elif translation_label in ["b", "ne"]:
            x_vals, y_vals = AsteriskPlotting.get_b(n_samples)
        elif translation_label in ["c", "ea"]:
            x_vals, y_vals = AsteriskPlotting.get_c(n_samples)
        elif translation_label in ["d", "se"]:
            x_vals, y_vals = AsteriskPlotting.get_d(n_samples)
        elif translation_label in ["e", "so"]:
            x_vals, y_vals = AsteriskPlotting.get_e(n_samples)
        elif translation_label in ["f", "sw"]:
            x_vals, y_vals = AsteriskPlotting.get_f(n_samples)
        elif translation_label in ["g", "we"]:
            x_vals, y_vals = AsteriskPlotting.get_g(n_samples)
        elif translation_label in ["h", "nw"]:
            x_vals, y_vals = AsteriskPlotting.get_h(n_samples)
        elif translation_label in ["n", "x"]:
            x_vals, y_vals = 0, 0  # want to rotate around center point
        else:
            # throw error
            pass

        return x_vals, y_vals

    @staticmethod
    def plot_all_target_lines(specific_lines=None):
        """  # TODO: adapt for new t labels
        Plot all target lines on a plot for easy reference
        :param order_of_colors:
        """
        if specific_lines is None:
            x_a, y_a = AsteriskPlotting.get_a()
            x_b, y_b = AsteriskPlotting.get_b()
            x_c, y_c = AsteriskPlotting.get_c()
            x_d, y_d = AsteriskPlotting.get_d()
            x_e, y_e = AsteriskPlotting.get_e()
            x_f, y_f = AsteriskPlotting.get_f()
            x_g, y_g = AsteriskPlotting.get_g()
            x_h, y_h = AsteriskPlotting.get_h()

            ideal_xs = [x_a, x_b, x_c, x_d, x_e, x_f, x_g, x_h]
            ideal_ys = [y_a, y_b, y_c, y_d, y_e, y_f, y_g, y_h]

            dirs = datamanager.get_option_list("translation_only")
            for i, d in enumerate(dirs):
                plt.plot(ideal_xs[i], ideal_ys[i], color=AsteriskPlotting.get_dir_color(d),
                         label='ideal', linestyle='--', zorder=10)

        else:  # there are specific directions you want to plot, and only those directions
            ideal_xs = list()
            ideal_ys = list()

            if "a" in specific_lines or "no" in specific_lines:
                x_a, y_a = AsteriskPlotting.get_a()
                ideal_xs.append(x_a)
                ideal_ys.append(y_a)

            if "b" in specific_lines or "ne" in specific_lines:
                x_b, y_b = AsteriskPlotting.get_b()
                ideal_xs.append(x_b)
                ideal_ys.append(y_b)

            if "c" in specific_lines or "ea" in specific_lines:
                x_c, y_c = AsteriskPlotting.get_c()
                ideal_xs.append(x_c)
                ideal_ys.append(y_c)

            if "d" in specific_lines or "se" in specific_lines:
                x_d, y_d = AsteriskPlotting.get_d()
                ideal_xs.append(x_d)
                ideal_ys.append(y_d)

            if "e" in specific_lines or "so" in specific_lines:
                x_e, y_e = AsteriskPlotting.get_e()
                ideal_xs.append(x_e)
                ideal_ys.append(y_e)

            if "f" in specific_lines or "sw" in specific_lines:
                x_f, y_f = AsteriskPlotting.get_f()
                ideal_xs.append(x_f)
                ideal_ys.append(y_f)

            if "g" in specific_lines or "we" in specific_lines:
                x_g, y_g = AsteriskPlotting.get_g()
                ideal_xs.append(x_g)
                ideal_ys.append(y_g)

            if "h" in specific_lines or "nw" in specific_lines:
                x_h, y_h = AsteriskPlotting.get_h()
                ideal_xs.append(x_h)
                ideal_ys.append(y_h)

            for i, dir in enumerate(specific_lines):
                plt.plot(ideal_xs[i], ideal_ys[i], color=AsteriskPlotting.get_dir_color(dir),
                         label='ideal', linestyle='--', zorder=10)

    @staticmethod
    def plot_notes(labels, ax):  # TODO: move to aplt, make it take in a list of labels so HandTranslation can also use it
        """
        Plots the labels and trial ID in the bottom left corner of the plot
        """
        note = "Labels in plotted data:"

        #labels = set()
        # for a in trials:  # self.averages:
        #     for l in a.trialset_labels:
        #         labels.add(l)

        for l in list(labels):
            note = f"{note} {l} |"

        #ax = plt.gca()
        # plt.text(0.1, 0.2, self.generate_name()) #, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))
        ax.text(-0.1, -0.12, note, transform=ax.transAxes) #, bbox=dict(facecolor='blue', alpha=0.5))

    @staticmethod
    def add_dist_label(atrial, ax=None):
        """
        Makes a text object appear at the head of the target line
        """
        modifiers = dict(a=(0, 1), b=(1, 1), c=(1, 0), d=(1, -1), e=(0, -1), f=(-1, -1), g=(-1, 0), h=(-1, 1),
                         no=(0, 1), ne=(1, 1), ea=(1, 0), se=(1, -1), so=(0, -1), sw=(-1, -1), we=(-1, 0), nw=(-1, 1))

        #for t in trial:
        xt, yt = AsteriskPlotting.get_direction(atrial.trial_translation, n_samples=2)
        # print(f"{atrial.trial_translation} => [{xt[1]}, {yt[1]}]")

        # get the spacing just right for the labels
        if atrial.trial_translation in ["b", "d", "f", "h", "ne", "se", "sw", "nw"]:
            x_plt = xt[1] + np.abs(xt[1]) * 0.1 * modifiers[atrial.trial_translation][0] + 0.05 * modifiers[atrial.trial_translation][0]
            y_plt = yt[1] + np.abs(yt[1]) * 0.1 * modifiers[atrial.trial_translation][1]
        elif atrial.trial_translation in ["a", "c", "e", "g", "no", "so", "ea", "we"]:
            x_plt = xt[1] + np.abs(xt[1]) * 0.1 * modifiers[atrial.trial_translation][0] + 0.03 * modifiers[atrial.trial_translation][0]
            y_plt = yt[1] + np.abs(yt[1]) * 0.1 * modifiers[atrial.trial_translation][1] + 0.03 * modifiers[atrial.trial_translation][1]
        else:
            print("Else triggered incorrectly.")
            x_plt = xt[1] + np.abs(xt[1]) * 0.1 * modifiers[atrial.trial_translation][0]
            y_plt = yt[1] + np.abs(yt[1]) * 0.1 * modifiers[atrial.trial_translation][1]

        # print(f"{atrial.trial_translation} => [{x_plt}, {y_plt}]")
        translate_labels = {
            "no": "N",
            "ne": "NE",
            "nw": "NW",
            "ea": "E",
            "we": "W",
            "so": "S",
            "se": "SE",
            "sw": "SW",
            "a": "N",
            "b": "NE",
            "h": "NW",
            "c": "E",
            "g": "W",
            "e": "S",
            "d": "SE",
            "f": "SW"
        }

        ax.text(x_plt, y_plt, f"{translate_labels[atrial.trial_translation]}: {np.round(atrial.total_distance, 2)}",
                style='italic', ha='center', va='center'
                #bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2}
                )

    @staticmethod
    def add_obj_img(file_loc_obj, rotation, fig):
        """
        Plots a small image which illustrates the AstHandTranslation.set_rotation value
        """
        n_loc = file_loc_obj.resources / "cube_n.jpg"
        m_loc = file_loc_obj.resources / "cube_m15.jpg"
        p_loc = file_loc_obj.resources / "cube_p15.jpg"
        img_locs = dict(n=n_loc, m15=m_loc, p15=p_loc,
                        x=n_loc)

        im = plt.imread(img_locs[rotation])
        newax = fig.add_axes([0.07, 0.86, 0.12, 0.12], anchor='NW', zorder=0)
        newax.imshow(im)
        newax.axis('off')

    # TODO: more functions: plot_asterisk, plot_avg_asterisk, plot_one_direction -> move this here and out of the other objects

    @staticmethod
    def plot_a_trial(file_loc, trial,
                     use_filtered=True, include_notes=False, labels=None, plot_orientations=False,
                     incl_obj_img=True, save_plot=False, show_plot=True):
        """
        Plots a single trial
        """

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)

        ideal_x, ideal_y = AsteriskPlotting.get_direction(trial.trial_translation)
        ax.plot(ideal_x, ideal_y, color=AsteriskPlotting.get_dir_color(trial.trial_translation),
                label='ideal', linestyle='--', zorder=30)

        data_x, data_y, _ = trial.get_poses(use_filtered)
        ax.plot(data_x, data_y, color="xkcd:dark blue", label='trajectory')

        max_x = max(data_x)
        max_y = max(data_y)
        min_x = min(data_x)
        min_y = min(data_y)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path of Object')

        # gives a realistic view of what the path looks like
        plt.xticks(np.linspace(AsteriskPlotting.round_half_down(min_x, decimals=2),
                               AsteriskPlotting.round_half_up(max_x, decimals=2), 10), rotation=30)

        if trial.trial_translation in ["a", "b", "c", "g", "h", "no", "so", "ea", "we"]:
            plt.yticks(np.linspace(0, AsteriskPlotting.round_half_up(max_y, decimals=2), 10))
        else:
            plt.yticks(np.linspace(AsteriskPlotting.round_half_down(min_y, decimals=2), 0, 10))

        # plt.gca().set_aspect('equal', adjustable='box')

        plt.title(f"Plot: {trial.generate_plot_title()}")

        if plot_orientations:
            trial._plot_orientations(scale=1.0)

        if incl_obj_img:
            AsteriskPlotting.add_obj_img(file_loc, trial.trial_rotation, fig)

        if include_notes:
            trial._plot_notes()

        if show_plot:
            plt.legend()
            plt.show()

        if save_plot:
            plt.savefig(file_loc.result_figs / f"plot_{trial.generate_name()}.jpg", format='jpg')
            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        return ax

    @staticmethod
    def compare_paths(first_path, second_path):
        """
        Takes two paths (assuming the same direction) and plots them on top of each other.
        Also plots the ideal line, taken from the first trial obj
        """
        colors = ["black", "red", "blue"]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)

        plot_direction = first_path.trial_translation

        ideal_x, ideal_y = AsteriskPlotting.get_direction(plot_direction)

        # plot ideal line
        ax.plot(ideal_x, ideal_y, color=colors[0], label=f"ideal line, dir {plot_direction}", zorder=10)

        # plot first path
        first_path.moving_average()
        first_x, first_y, _ = first_path.get_poses(use_filtered=True)
        first_name = first_path.generate_name()
        ax.plot(first_x, first_y, color=colors[1], label=f"path 1: {first_name}", zorder=15)

        # plot second path
        second_path.moving_average()
        second_x, second_y, _ = second_path.get_poses(use_filtered=True)
        second_name = second_path.generate_name()
        ax.plot(second_x, second_y, color=colors[2], label=f"path 2: {second_name}", zorder=20)

        ax.set_title(f"Dir {plot_direction} comparison")
                
        plt.legend()
        plt.show()

    @staticmethod
    def plot_sd(axes, avg_trial, color, use_filtered=True):

        avg_x, avg_y, _ = avg_trial.get_poses(use_filtered=use_filtered)

        ad_x_up, ad_y_up, _ = avg_trial.get_poses_ad(which_set=1)
        ad_x_down, ad_y_down, _ = avg_trial.get_poses_ad(which_set=2)

        # necessary for building the polygon
        r_ad_x = list(reversed(ad_x_down))
        r_ad_y = list(reversed(ad_y_down))

        poly = []
        for ax, ay in zip(ad_x_up, ad_y_up):
            pt = [ax, ay]
            poly.append(pt)

        # add last point for nicer looking plot
        last_pose = avg_trial.get_last_pose()
        poly.append([last_pose[0], last_pose[1]])

        for ax, ay in zip(r_ad_x, r_ad_y):
            pt = [ax, ay]
            poly.append(pt)

        polyg = plt.Polygon(poly, color=color, alpha=0.4)
        #plt.gca().add_patch(polyg)
        axes.add_patch(polyg)

    @staticmethod
    def plot_asterisk(file_loc, dict_of_trials, rotation_condition="x", hand_name="",
                      #plotting_averages_with_sd=False,
                      use_filtered=True, linestyle="solid",
                      include_notes=False, labels=None,
                      plot_orientations=False, tdist_labels=True,
                      incl_obj_img=True, gray_it_out=False,
                      save_plot=False, show_plot=True):
        """
        Takes a dictionary of trials
        (key -> label of direction, value -> list of trials you want to plot
        """
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)

        AsteriskPlotting.plot_all_target_lines()

        for dir in dict_of_trials.keys():
            for t in dict_of_trials[dir]:
                data_x, data_y, theta = t.get_poses(use_filtered)

                # ax.plot(data_x, data_y, color=colors[i], label='trajectory', linestyle=linestyle)
                if gray_it_out:  # TODO: make a gradient of grey, so that we can tell what direction is what
                    ax.plot(data_x, data_y, color="lightgrey",
                            label='trajectory', linestyle=linestyle, zorder=30)
                else:
                    ax.plot(data_x, data_y, color=AsteriskPlotting.get_dir_color(t.trial_translation),
                            label='trajectory', linestyle=linestyle, zorder=30)

                if plot_orientations:
                    t._plot_orientations(marker_scale=15, line_length=0.025, scale=1)

                # plot total_distance value in each direction
                if tdist_labels:
                    AsteriskPlotting.add_dist_label(t, ax=ax)

        if include_notes and labels is not None:
            # TODO: revisit this
            AsteriskPlotting.plot_notes(labels, ax=ax)

        if incl_obj_img:
            AsteriskPlotting.add_obj_img(file_loc, rotation_condition, fig)

        fig.suptitle(f"{hand_name}, {rotation_condition} Avg Asterisk", fontweight="bold", fontsize=14)
        ax.set_title("Cube size: ~0.25 span, init pos: 0.75 depth")  # , pad=10)
        # plt.title(f"{self.hand.get_name()} avg asterisk")  # , rot: {trials[0].trial_rotation}")
        ax.axis([-0.7, 0.7, -0.7, 0.7])
        ax.tick_params(axis="x", rotation=30)
        # plt.xticks(np.linspace(-0.7, 0.7, 15), rotation=30)
        # plt.yticks(np.linspace(-0.7, 0.7, 15))
        plt.gca().set_aspect('equal', adjustable='box')

        # TODO: add in the average deviation regions
        # if plotting_averages_with_sd:
        #     for a_k in list(dict_of_trials.keys()):
        #         a_trial = dict_of_trials[a_k][0]  # if plotting averages with sd, we can assume there is only one object
        #         AsteriskPlotting.plot_sd(ax, a_trial, AsteriskPlotting.get_dir_color(a_trial.trial_translation))

        if show_plot:
            # plt.legend()  # TODO: showing up weird, need to fix
            plt.show()

        if save_plot:  # TODO: how will I do file locations here?
            plt.savefig(file_loc.result_figs / f"ast_{hand_name}_{rotation_condition}.jpg", format='jpg')
            #plt.savefig(f"results/pics/avgd_{self.hand.get_name()}_{len(self.subjects_containing)}subs_{self.set_rotation}.jpg", format='jpg')

            # name -> tuple: subj, hand  names
            print("Figure saved.")
            print(" ")

        return ax, fig

