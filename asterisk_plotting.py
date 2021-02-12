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
