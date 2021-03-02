import math as m
import numpy as np


class AsteriskPlotting:
    """
    Resources for asterisk plots
    """

    def __init__(self):
        pass

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
    def straight(num_points=25, mod=1):
        vals = np.linspace(0, 0.5, num_points)
        z = np.zeros(num_points)

        set1 = mod * vals
        return set1, z

    @staticmethod
    def diagonal(num_points=25, mod1=1, mod2=1):
        coords = np.linspace(0, 0.5, num_points)

        set1 = mod1 * coords
        set2 = mod2 * coords

        return set1, set2

    @staticmethod
    def get_a(num_points=25):
        y_coords, x_coords = AsteriskPlotting.straight(num_points)
        return x_coords, y_coords

    @staticmethod
    def get_b(num_points=25):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points)
        return x_coords, y_coords

    @staticmethod
    def get_c(num_points=25):
        x_coords, y_coords = AsteriskPlotting.straight(num_points)
        return x_coords, y_coords

    @staticmethod
    def get_d(num_points=25):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points=num_points, mod1=1, mod2=-1)
        return x_coords, y_coords

    @staticmethod
    def get_e(num_points=25):
        y_coords, x_coords = AsteriskPlotting.straight(num_points, mod=-1)
        return x_coords, y_coords

    @staticmethod
    def get_f(num_points=25):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points=num_points, mod1=-1, mod2=-1)
        return x_coords, y_coords

    @staticmethod
    def get_g(num_points=25):
        x_coords, y_coords = AsteriskPlotting.straight(num_points, mod=-1)
        return x_coords, y_coords

    @staticmethod
    def get_h(num_points=25):
        x_coords, y_coords = AsteriskPlotting.diagonal(num_points, mod1=-1)
        return x_coords, y_coords

    @staticmethod
    def get_direction(translation_label, n_samples=100):
        """
        return the appropriate x and y coordinates for any direction
        """
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
