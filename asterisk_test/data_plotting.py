import math as m
import numpy as np
from matplotlib import pyplot as plt



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
                  "e": "tab:cyan", "f": "tab:green", "g": "tab:pink", "h": "tab:orange"}
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
        """
        return the appropriate x and y coordinates for any direction
        """
        # TODO: make it so we can customize the amount in each direction
        x_vals, y_vals = 0, 0
        if translation_label == "a":
            x_vals, y_vals = AsteriskPlotting.get_a(n_samples)
        elif translation_label == "b":
            x_vals, y_vals = AsteriskPlotting.get_b(n_samples)
        elif translation_label == "c":
            x_vals, y_vals = AsteriskPlotting.get_c(n_samples)
        elif translation_label == "d":
            x_vals, y_vals = AsteriskPlotting.get_d(n_samples)
        elif translation_label == "e":
            x_vals, y_vals = AsteriskPlotting.get_e(n_samples)
        elif translation_label == "f":
            x_vals, y_vals = AsteriskPlotting.get_f(n_samples)
        elif translation_label == "g":
            x_vals, y_vals = AsteriskPlotting.get_g(n_samples)
        elif translation_label == "h":
            x_vals, y_vals = AsteriskPlotting.get_h(n_samples)
        elif translation_label == "n":
            x_vals, y_vals = 0, 0  # want to rotate around center point
        else:
            # throw error
            pass

        return x_vals, y_vals

    # TODO: more functions: plot_asterisk, plot_avg_asterisk, plot_one_direction -> move this here and out of the other objects

    @staticmethod
    def compare_paths(first_path, second_path):
        """
        Takes two paths (assuming the same direction) and plots them on top of each other.
        Also plots the ideal line, taken from the first trial obj
        """
        colors = ["black", "red", "blue"]

        plot_direction = first_path.trial_translation

        ideal_x, ideal_y = AsteriskPlotting.get_direction(plot_direction)

        # plot ideal line
        plt.plot(ideal_x, ideal_y, color=colors[0], label=f"ideal line, dir {plot_direction}")
        plt.title(f"Dir {plot_direction} comparison")

        # plot first path
        first_x, first_y, _ = first_path.get_poses()
        first_name = first_path.generate_name()
        plt.plot(first_x, first_y, color=colors[1], label=f"path 1: {first_name}")

        # plot second path
        second_x, second_y, _ = second_path.get_poses()
        second_name = second_path.generate_name()
        plt.plot(second_x, second_y, color=colors[2], label=f"path 2: {second_name}")

