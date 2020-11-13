"""
This file will...
0) generate the ideal line

"""

import csv
import pandas as pd
import numpy as np
import asterisk_0_prompts as prompts

#well, what I can do is do a linspace for both x and y... its straightforward because these are perfect lines we are drawing

def straight(num_points=11, mod=1):
    vals = np.linspace(0,0.5,num_points)
    z = [0.] * num_points

    set1 = mod * vals
    return set1, z

def diagonal(num_points=11, mod1=1, mod2=1):
    coords = np.linspace(0,0.5,num_points)

    set1 = mod1 * coords
    set2 = mod2 * coords

    return set1, set2

def get_a(num_points=11):
    y_coords, x_coords = straight(num_points)
    return x_coords, y_coords

def get_b(num_points=11):
    x_coords, y_coords = diagonal(num_points)
    return x_coords, y_coords

def get_c(num_points=11):
    x_coords, y_coords = straight(num_points)
    return x_coords, y_coords

def get_d(num_points=11):
    x_coords, y_coords = diagonal(num_points=num_points, mod2=-1)
    return x_coords, y_coords

def get_e(num_points=11):
    y_coords, x_coords = straight(num_points, mod=-1)
    return x_coords, y_coords

def get_f(num_points=11):
    x_coords, y_coords = diagonal(num_points=num_points, mod1=-1, mod2=-1)
    return x_coords, y_coords

def get_g(num_points=11):
    x_coords, y_coords = straight(num_points, mod=-1)
    return x_coords, y_coords

def get_h(num_points=11):
    x_coords, y_coords = diagonal(num_points, mod1=-1)
    return x_coords, y_coords