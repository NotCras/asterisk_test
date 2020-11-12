"""
This file will...
0) plot object path and orientation
1) makes an image of the plot
2) has the option to plot a single file or multiple directions

TODO: what to do about combining multiple trials of a single direction? Should that be here? Should it be in its own script?

"""

import csv
import numpy as np
import pandas as pd
import math as m
import asterisk_0_prompts as prompts
import matplotlib.pyplot as plt


#from: https://realpython.com/python-rounding/
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return m.floor(n*multiplier + 0.5) / multiplier

#from: https://realpython.com/python-rounding/
def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return m.ceil(n*multiplier - 0.5) / multiplier


def get_data(path_to_data):
    df = pd.read_csv(total_path, 
        #names=["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"],
        skip_blank_lines=True
    )

    df_numeric = df.apply( pd.to_numeric )

    df_numeric.columns = ["row", "x", "y", "rmag", "f_x", "f_y", "f_rot_mag"]
    #df_numeric = pd.to_numeric(df) #data comes in as a str?

    return df_numeric

def plot_data(df):
    data_x = pd.Series.to_list(df["f_x"])
    data_y = pd.Series.to_list(df["f_y"])
    theta = pd.Series.to_list(df["f_rot_mag"])

    plt.plot(data_x, data_y, color='crimson', label='trajectory')

    #plot data points separately to show angle error with marker size
    for n in range(len(data_x)):
        plt.plot(data_x[n], data_y[n], 'r.', alpha=0.5, markersize=5*theta[n]) #rn having difficulty doing marker size in a batch, so plotting each point separately 
    
    max_x = max(data_x)
    max_y = max(data_y)
    min_x = min(data_x)
    #plt.scatter(data_x, data_y, marker='o', color='red', alpha=0.5, s=5*theta)

    print(f"max_x: {max_x}, min_x: {min_x}, y: {max_y}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path of Object')
    #plt.grid()
    plt.savefig(file_name + "plotted")

    
    plt.xticks(np.linspace(round_half_down(min_x, decimals=2), round_half_up(max_x, decimals=2), 10), rotation=30)
    plt.yticks(np.linspace(0, round_half_up(max_y, decimals=2), 10))

    #plt.xlim(0., 0.5)
    #plt.ylim(0., 0.5)
    plt.show()


if __name__ == "__main__":
    folder_path = "filtered/"
    #TODO: Add a prompt for plotting different data
    file_name = "filt_" + "josh_2v2_a_none_1" 
    total_path = folder_path + file_name + ".csv"
    print(total_path)

    df = get_data(total_path)

    plot_data(df)
