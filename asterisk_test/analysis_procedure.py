
import numpy as np
import logging as log
from pathlib import Path
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from ast_hand_translation import AstHandTranslation
from ast_hand_rotation import AstHandRotation
from file_manager import AstDirectory
from ast_iterator import AstIterator, my_option_dict
from aruco_analysis import AstArucoAnalysis
from metric_analyzers import AstHandAnalyzer


"""
Step 1: make and populate an AstDirectory File
"""
saved_file_directory = None

if saved_file_directory is None:
    home_directory = Path(__file__).parent.absolute()
    data_directory = home_directory / "data"

    # put together an AstDirectory manually
    ast_files = AstDirectory()
    ast_files.compressed_data = data_directory / "compressed_data"
    ast_files.aruco_pics = data_directory / "viz"
    ast_files.aruco_data = data_directory / "aruco_data"
    ast_files.path_data = data_directory / "trial_paths"
    ast_files.metric_results = data_directory / "results"
    ast_files.result_figs = data_directory / "results" / "plots"
    ast_files.debug_figs = data_directory / "results" / "debug_plots"

else: 
    # if we have a file, we can load it automatically

    #TODO: add this
    raise NotImplementedError("Can't load a file directory by file yet!")

ast_options = AstIterator(my_option_dict)

"""
Step 2: Set up the process
"""
# pertaining to what data to include
included_hands = []
included_subjects = []
included_trials_file = None
blocked_trials_file = None
do_rotation_only_trials = False
do_rotation_conditions_for_tr = False
included_rotation_conditions = []

# pertaining to data generation
normalize_data = True
run_aruco_analysis = True
run_averaging_calculation = True
save_data = True
save_plots = True
save_avg_debug_plots = True

# pertaining to metric calculation
run_metric_calculation = True
included_metrics = []
save_metrics = True

# pertaining to terminal spam
show_verbose_logs = False

# number of trial sets, used for progress bar
# TODO: add improved calculation here
"""
H_tr * (len(do_rotation_conditions_for_tr) + do_rotation_only_trials) + H_to
H_tr -> hands that can do rotation
H_to -> hands that cannot do rotation
"""
tr_hands = None
to_hands = None
total_bars = 1 

# TODO: so far, things are set up for translation only trials... need to revisit for rotation_only and tr
with alive_bar(total_bars) as bar:
    for h in included_hands:
        log.info(f"Running: {h}, {included_subjects}")

        """ Now for each set of trial data...
        Step 3: run aruco analysis, if desired
        """
        ar_data = None
        if run_aruco_analysis:
            log.info("======  Running aruco analysis")
            mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                            (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                            (0, 0, 1)))
            dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))

            marker_side_dims = 0.03 # in meters

            ar = AstArucoAnalysis(ast_files, mtx, dists, marker_side_dims)

            ar_data = []
            for s in included_subjects: # TODO: how should I separate between the rotation_only trials, the tr trials, and the t_only trials?
                ar_analysis = ar.batch_aruco_analysis(s, h, 
                                                    include_rotation_only_trials=do_rotation_only_trials, 
                                                    exclude_tr_trials=do_rotation_conditions_for_tr,  
                                                    save_data=False, assess_indices=False, crop_trial=False)

                ar_data.append(ar_analysis) 

        """
        Step 4: Data conditioning and organizing, plotting paths
        """
        log.info("=====  Running data conditioning and organizing, plotting paths")

        data = AstHandTranslation(subjects, h, rotation=rot, blocklist_file="trial_blocklist.csv")

        if ar_data is None:
            # If we didn't run the aruco analysis, then we need to import the aruco data
            data.get_data_from_filenames() 

        else:
            # Otherwise, we take the aruco data we just generated
            data.get_data_from_arucolocs(ar_data) 

        data.filter_data(10)


        # make plots... a) averaged plot, if desired, b) average debug plots, if desired, and c) is there anything else?
        data.plot_ast_avg(show_plot=False, save_plot=True, exclude_path_labels=['major deviation'])

        if save_avg_debug_plots:
            for a in data.averages:
                a.avg_debug_plot(show_plot=False, save_plot=True, use_filtered=True)

        # although we don't show the plots, a matplotlib warning suggests that it still keeps those plots open
        plt.close("all")

        # TODO: I should save plots both as jpgs and as svgs (in separate folders!)

        """
        Step 5: Metric calculation, if desired
        """
        if run_metric_calculation:
            log.info("=====  Running metric calculation")
            raise NotImplementedError("Can't run metric calculation yet.")

            metric_data = AstHandAnalyzer(data)


        """
        Step 6: Saving data (aruco, path data, metric results), if desired
        """
        if run_aruco_analysis:
            log.info("Saving aruco data.")
            for a_data in ar_data:
                a_hand = a_data.data_attributes["hand"]
                a_t = a_data.data_attributes["translation"]
                a_rot = a_data.data_attributes["rotation"]
                a_sub = a_data.data_attributes["subject"]
                a_num = a_data.data_attributes["trial_num"]

                a_data.save_poses(file_name_overwrite=f"aruco_{a_hand}_{a_t}_{a_rot}_{a_sub}_{a_num}")

        log.info("Saving path data.")
        data.save_all_data()

        if run_metric_calculation:
            log.info("Saving metric results.")
            metric_data.save_data(file_name_overwrite=f"{h}_{rot}")


        """
        Now we repeat for the next set of trial data... 
        """