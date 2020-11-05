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

folder_path = "filtered/"
file_name = "filt_" + "josh_2v2_a_none_1" + ".csv"

total_path = folder_path + file_name
print(total_path)

df = pd.read_csv(total_path, names=[
    "x", "y", "rmag", "f_x", "f_y","f_rot_mag"],
    skip_blank_lines=True,
    )

df.dropna()

#plt.scatter([1.5,2.1,3.2], [1.75,2.0,3.9], marker='o', s=[5,30,60])
print(df["f_x"])
plt.plot(df["f_x"].to_numpy(), df["f_y"].to_numpy())#, marker='o', s=df["f_rot_mag"].to_numpy() )

plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(0, 100)
#plt.ylim(0, 5)
plt.show()
