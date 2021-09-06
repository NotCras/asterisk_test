import matplotlib.pyplot as plt
import numpy as np
from ast_hand_translation import AstHandTranslation


class AstHandRotation(AstHandTranslation):
    def __init__(self):
        pass

    # TODO: this takes in trial rotations for cw and ccw each

    # TODO: go through AstHandTranslation (the soon to be renamed AstHandTrials class)

    # TODO: need to redo averaging function?

    # TODO: replace all target lines function to better serve rotation plot?

    def plot_ast_avg(self, subjects=None, show_plot=True, save_plot=False, include_notes=True,
                     linestyle="solid", plot_contributions=False, exclude_path_labels=None):
        # TODO: this is the test code that I got working for this, but now we need to get it working for this object
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        # data = [float(x.split()[0]) for x in recipe]

        rotation = 78
        rotation2 = 57
        data = [rotation, rotation2, 360 - rotation - rotation2]

        ingredients = [f"{str(rotation)}{chr(176)}", f"{str(rotation2)}{chr(176)}", ""]  # "not rotated"]

        colors = ["crimson", "royalblue", "whitesmoke"]

        wedges, texts = ax.pie(data, colors=colors, labels=ingredients, labeldistance=0.75,
                               startangle=90 - rotation, counterclock=True,
                               textprops=dict(color="whitesmoke", size=12, weight="bold",
                                              # rotation_mode = 'anchor', va='center', ha='center'
                                              ))

        # draw circle by drawing a white circle on top of center
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        ax.set_title("Trial Max Rotation")

        # translation of the trial
        line_x = [0, 0.1, 0.2, 0.3]
        line_y = [0, 0.05, 0.1, 0.08]
        plt.plot(line_x, line_y, color="crimson")

        line2_x = [0, 0.05, 0.1, 0.08]
        line2_y = [0, 0.1, 0.2, 0.3]
        plt.plot(line2_x, line2_y, color="royalblue")

        # plot target lines
        target_line_dist = 0.5

        line_a_x = [0, 0]
        line_a_y = [0, target_line_dist]

        line_e_x = [0, 0]
        line_e_y = [0, -target_line_dist]

        line_c_x = [0, target_line_dist]
        line_c_y = [0, 0]

        line_g_x = [0, -target_line_dist]
        line_g_y = [0, 0]

        lines_x = [line_a_x, line_c_x, line_e_x, line_g_x]
        lines_y = [line_a_y, line_c_y, line_e_y, line_g_y]

        for x, y in zip(lines_x, lines_y):
            plt.plot(x, y, color='lightsteelblue', linestyle="--")

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        # plt.tight_layout()
        plt.show()
