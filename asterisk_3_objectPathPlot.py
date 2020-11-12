"""
This file will...
0) plot object path and orientation
1) makes an image of the plot
2) has the option to plot a single file or multiple directions

TODO: what to do about combining multiple trials of a single direction? Should that be here? Should it be in its own script?

"""

import csv
import pandas as pd
import numpy as np
import asterisk_0_prompts as prompts
import matplotlib.pyplot as plt

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

    #plt.plot(data_x, data_y, marker='o', markersize=theta)
    max_x = 0
    max_y = 0
    min_x = 0
    plt.plot(data_x, data_y, color='crimson', label='trajectory')

    #plot data points separately to show angle error with marker size
    for n in range(len(data_x)):
        max_x = max(data_x[n], max_x)
        max_y = max(data_y[n], max_y)

        min_x = min(data_x[n], min_x)
        plt.plot(data_x[n], data_y[n], 'r.', markersize=5*theta[n])

    print(f"max_x: {max_x}, min_x: {min_x}, y: {max_y}")

    plt.xlabel('x path')
    plt.ylabel('y path')
    plt.title('Path of Object')
    #plt.grid()
    plt.savefig(file_name + "plotted")

    plt.xticks(np.linspace(min_x, max_x, 10), rotation=45 )
    plt.yticks(np.linspace(0, max_y, 10) )

    #plt.xlim(0., 0.5)
    #plt.ylim(0., 0.5)
    plt.show()


if __name__ == "__main__":
    folder_path = "filtered/"
    file_name = "filt_" + "josh_2v2_a_none_1" 
    total_path = folder_path + file_name + ".csv"
    print(total_path)

    df = get_data(total_path)

    plot_data(df)
