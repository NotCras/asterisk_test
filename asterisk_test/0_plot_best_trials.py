import pdb
import matplotlib.pyplot as plt

from ast_trial_translation import AstTrialTranslation
from trial_helper import read_best_trials
from file_manager import my_ast_files
from data_plotting import AsteriskPlotting as aplt

num_best = 1
best_trial_dict = read_best_trials(f"top_trials{num_best}_all.csv")
print(f"Reading: top trials {num_best}")

for h in ["2v2", "2v3", "3v3", "m2active", "palm1r", "palm2r"]:
    for r in ["n", "m15", "p15"]:
        plot_trials_dict = {"a": [], "b": [], "c": [], "d": [], "e": [], "f": [], "g": [], "h": []}

        if r in ["m15", "p15"] and h in ["m2active", "palm1r"]:
            print(f"Skipping {h}_{r}, best trials only")
        else:
            print(f"Now plotting {h}_{r}, best trials only")
            for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                label = f"{h}_{t}_{r}"

                # get the best_trial for that label
                best_trial_name = best_trial_dict[label]

                best_trial = AstTrialTranslation(my_ast_files)

                # get the ast_trial read
                print(f"Best trial for {label}: {best_trial_name}")

                h, t, r, s, n = best_trial_name.split("_")
                best_name_old_way = f"{s}_{h}_{t}_{r}_{n}"

                best_trial.load_data_by_aruco_file(best_name_old_way + ".csv")
                best_trial.moving_average(window_size=15)

                # insert it into the dict
                plot_trials_dict[t] = [best_trial]
                #pdb.set_trace()


            # make the asterisk plot!
            aplt.plot_asterisk(my_ast_files, dict_of_trials=plot_trials_dict,
                               rotation_condition=r, hand_name=h, use_filtered=True,
                               save_plot=True, show_plot=False, incl_obj_img=True)

# now, let's show all of the trials, and then plot the best trials on top
for h in ["2v2", "2v3", "3v3", "m2active", "palm1r", "palm2r"]:
    for r in ["n", "m15", "p15"]:
            plot_trials_dict = {"a": [], "b": [], "c": [], "d": [], "e": [], "f": [], "g": [], "h": []}
            plot_all_dict = {"a": [], "b": [], "c": [], "d": [], "e": [], "f": [], "g": [], "h": []}


            if r in ["m15", "p15"] and h in ["m2active", "palm1r"]:
                print(f"Skipping {h}_{r}")
            else:
                print(f"Now plotting {h}_{r}")
                for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    label = f"{h}_{t}_{r}"

                    # get the best_trial for that label
                    best_trial_name = best_trial_dict[label]

                    best_trial = AstTrialTranslation(my_ast_files)

                    # get the ast_trial read
                    print(f"Best trial for {label}: {best_trial_name}")

                    h, t, r, s, n = best_trial_name.split("_")
                    best_name_old_way = f"{s}_{h}_{t}_{r}_{n}"

                    best_trial.load_data_by_aruco_file(best_name_old_way + ".csv")
                    best_trial.moving_average(window_size=15)

                    # insert it into the dict
                    plot_trials_dict[t] = [best_trial]
                    #pdb.set_trace()

                    for s in ["sub1", "sub2", "sub3"]:
                        for n in ["1", "2", "3", "4", "5"]:
                            try:
                                trial = AstTrialTranslation(my_ast_files)
                                trial_name = f"{s}_{h}_{t}_{r}_{n}"
                                trial.load_data_by_aruco_file(trial_name + ".csv")
                                trial.moving_average(window_size=15)
                                plot_all_dict[t].append(trial)
                            except:
                                print(f"Just so you know, this trial failed: {h}_{t}_{r}_{s}_{n}")

                    print(f"{t} -> {plot_all_dict[t]}")

                # make the asterisk plot!
                ax = aplt.plot_asterisk(my_ast_files, dict_of_trials=plot_all_dict, use_filtered=True,
                                   rotation_condition=r, hand_name=h, tdist_labels=False, gray_it_out=True,
                                   save_plot=False, show_plot=False, incl_obj_img=True)

                for t in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    best_trial = plot_trials_dict[t][0]
                    t_x, t_y, _ = best_trial.get_poses(use_filtered=True)
                    ax.plot(t_x, t_y, color=aplt.get_dir_color(best_trial.trial_translation),
                                label=f"best {t}", zorder=100)
                    aplt.add_dist_label(best_trial, ax=ax)

                plt.savefig(my_ast_files.result_figs / f"ast_all_{h}_{r}.jpg", format='jpg')

                # name -> tuple: subj, hand  names
                print("Figure saved.")
                print(" ")
