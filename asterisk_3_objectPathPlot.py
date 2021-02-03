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
from pathlib import Path
import asterisk_0_dataHelper as helper
import asterisk_0_prompts as prompts
import asterisk_4_idealLine as id
import matplotlib.pyplot as plt


#from: https://realpython.com/python-rounding/
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return m.floor(n*multiplier + 0.5) / multiplier

#from: https://realpython.com/python-rounding/
def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return m.ceil(n*multiplier - 0.5) / multiplier

def condition_df(df):
    df_numeric = df.apply( pd.to_numeric )

    df_numeric.columns = ["row", "x", "y", "rmag", "f_x", "f_y", "f_rot_mag"]
    
    return df_numeric

def get_data(path_to_data):
    df = pd.read_csv(path_to_data, 
        #names=["x", "y", "rmag", "f_x", "f_y", "f_rot_mag"],
        skip_blank_lines=True
    )

    df_numeric = condition_df(df)
    #df_numeric = pd.to_numeric(df) #data comes in as a str?

    return df_numeric

def data_from_df(df):
    x = pd.Series.to_list(df["f_x"])
    y = pd.Series.to_list(df["f_y"])
    ang = pd.Series.to_list(df["f_rot_mag"])

    return x, y, ang

def get_batch(sub, h, num):
    folder = "filtered/"
    file_root = f"filt_{sub}_{h}_"

    list_of_dfs = list()
    
    #if h == "basic" or h == "m2stiff" or h == "modelvf": 
    types = ["none"] #TODO: Add in a prompt for this one
    #else:
    #    types = prompts.type_options

    for t in types:
        #if t == "none": #TODO: add back in?
        #    directions = prompts.dir_options
        #else:
        directions = prompts.dir_options_no_rot #TODO: Not gonna plot rotation? Does this need fixing?

        for d in directions:
            #TODO: add more trial nums? TODO: Need to find a way to 'average' the trials
            file_name = file_root + d + "_" + t + "_" + num + ".csv"

            total_path = folder + file_name
            
            print("FILE: " + str(total_path))

            try:
                df = pd.read_csv(total_path)

            except:
                print("FAILED")
                continue

            df_numeric = condition_df(df)
        
            list_of_dfs.append(df_numeric)
            print(f"ADDED: {file_name}")

    return list_of_dfs

def plot_trial(df, file_name):
    data_x, data_y, theta = data_from_df(df)

    plt.plot(data_x, data_y, color='tab:red', label='trajectory')

    #plt.scatter(data_x, data_y, marker='o', color='red', alpha=0.5, s=5*theta)
    #plot data points separately to show angle error with marker size
    for n in range(len(data_x)):
        plt.plot(data_x[n], data_y[n], 'r.', alpha=0.5, markersize=5*theta[n]) #rn having difficulty doing marker size in a batch, so plotting each point separately 
    
    max_x = max(data_x)
    max_y = max(data_y)
    min_x = min(data_x)

    #print(f"max_x: {max_x}, min_x: {min_x}, y: {max_y}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path of Object')
    #plt.grid()
    
    plt.xticks(np.linspace(round_half_down(min_x, decimals=2), round_half_up(max_x, decimals=2), 10), rotation=30)
    #plt.xticks(np.linspace(0, round_half_up(max_y, decimals=2), 10), rotation=30) #gives a realistic view of what the path looks like
    plt.yticks(np.linspace(0, round_half_up(max_y, decimals=2), 10))

    #plt.xlim(0., 0.5)
    #plt.ylim(0., 0.5)

    plt.savefig("plot4_" + file_name + ".jpg", format='jpg')
    plt.show()

def plot_all_ideal(order_of_colors):
    print("PLOTTING IDEAL LINES")

    x_a, y_a = id.get_a()
    x_b, y_b = id.get_b()
    x_c, y_c = id.get_c()
    x_d, y_d = id.get_d()
    x_e, y_e = id.get_e()
    x_f, y_f = id.get_f()
    x_g, y_g = id.get_g()
    x_h, y_h = id.get_h()

    ideal_xs = [x_a, x_b, x_c, x_d, x_e, x_f, x_g, x_h]
    ideal_ys = [y_a, y_b, y_c, y_d, y_e, y_f, y_g, y_h]

    for i in range(8):
        plt.plot(ideal_xs[i], ideal_ys[i], color=order_of_colors[i], label='ideal', linestyle = '--')


def batch_plot(folder=None, sub=None, hand=None, num=None, show_plot=True):
    colors = ["tab:blue", "tab:purple", "tab:red",  "tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:orange"]

    if all(v is not None for v in [folder, sub, hand, num]):
        pass
    else:
        folder, sub, hand, num = prompts.request_name_hand_num_simple("filtered/")

    dfs = get_batch(sub, hand, num)

    #plot data
    for i,df in enumerate(dfs):
        data_x = pd.Series.to_list(df["f_x"])
        data_y = pd.Series.to_list(df["f_y"])
        theta = pd.Series.to_list(df["f_rot_mag"])

        plt.plot(data_x, data_y, color=colors[i], label='trajectory')  

        #plot data points separately to show angle error with marker size
        for n in range(len(data_x)):
            plt.plot(data_x[n], data_y[n], color=colors[i], alpha=0.5, markersize=10*theta[n])

    #plot ideal lines
    plot_all_ideal(colors)

    plt.xticks( np.linspace(-0.5, 0.5, 11), rotation=30 )
    plt.yticks( np.linspace(-0.5, 0.5, 11))

    plt.title(f"Plot: {sub}-{hand}, set #{num}")

    #saving figure
    plt.savefig(f"fullplot4_{sub}_{hand}_{num}.jpg", format='jpg') #name -> tuple: subj, hand  names
    print("Figure saved.")
    print(" ")
    if(show_plot):
        plt.show()


def auto_plot():
    folder_path, subject_name, hand = prompts.request_name_hand_simple("filtered/")

    #if hand == "basic" or hand == "m2stiff" or hand == "vf":
    #    types = ["none"]
    #else:
    #    types = prompts.type_options

    #for t in types: #TODO: add in a prompt for type of ast
        #if t == "none":
        #    directions = prompts.dir_options
        #else:
        #    directions = prompts.dir_options_no_rot

        #for d in directions:
    for num in ["1", "2", "3"]:
        batch_plot(folder=folder_path, sub=subject_name, hand=hand, num=num, show_plot=False)




def single_plot():
    folder_path = "filtered/"

    subject_name = helper.collect_prompt_data(
        prompts.subject_name_prompt, prompts.subject_name_options)

    hand = helper.collect_prompt_data(
        prompts.hand_prompt, prompts.hand_options)

    if hand == "basic" or hand == "m2stiff" or hand == "vf":
        types = ["none"]
    else:
        types = prompts.type_options

    t = helper.collect_prompt_data(
        prompts.type_prompt, types)

    if t == "none":
        directions = prompts.dir_options
    else:
        directions = prompts.dir_options_no_rot

    d = helper.collect_prompt_data(prompts.dir_prompt, directions)

    num = helper.collect_prompt_data(prompts.trial_prompt, prompts.trial_options)


    file_name = f"filt_{subject_name}_{hand}_{d}_{t}_{num}"
    total_path = folder_path + file_name + ".csv"
    print(total_path)

    df = get_data(total_path)

    plot_trial(df, file_name)

if __name__ == "__main__":
    home_directory = Path(__file__).parent.absolute()

    print("""
    
        ========= DATA PLOTTING ==========
      I PLOT YOUR DATA FOR THE ASTERISK STUDY
        AT NO COST, STRAIGHT TO YOUR DOOR!
    
    """)
    input("PRESS <ENTER> TO CONTINUE.  ")
    
    mode = input("Do you want to plot a batch of files? [y/n/auto] ")

    if mode == "y":
        batch_plot()
    elif mode == "auto":
        auto_plot()
    else:
        single_plot()
