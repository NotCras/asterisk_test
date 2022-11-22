from pathlib import Path
import numpy as np
import os

from file_manager import AstDirectory
from aruco_analysis import AstArucoAnalysis
from ast_hand_info import get_hand_stats

home_directory = Path("/home/john/Programs/new_ast_data")
data_directory = home_directory #/ "data"
new_ast_files = AstDirectory(home_directory)
new_ast_files.data_home = data_directory
new_ast_files.compressed_data = data_directory / "compressed_data"
new_ast_files.aruco_pics = data_directory / "viz"
new_ast_files.aruco_data = data_directory / "aruco_data"
new_ast_files.path_data = data_directory / "trial_paths"
new_ast_files.metric_results = data_directory / "results"
new_ast_files.result_figs = data_directory / "results" / "plots"
new_ast_files.debug_figs = data_directory / "results" / "debug_plots"
new_ast_files.resources = data_directory.parent / "resources"

mtx = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                (0, 0, 1)))
dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))

marker_side_dims = 0.03  # in meters

ar = AstArucoAnalysis(file_loc_obj=new_ast_files, camera_calib=mtx, camera_dists=dists, marker_side_dims=marker_side_dims)

# run aruco analysis on all the new data in the new location
viz_folders = [f for f in os.listdir(new_ast_files.aruco_pics) if f[0] != 'z']
# z indicates archived data that I don't want to use right now

for vf in viz_folders:
    print(vf)
    h, _, _, _, _ = vf.split("_")
    _, _, hand_id = get_hand_stats(h)

    ar.aruco_analyze_trial(trial_name=vf, aruco_id=hand_id, save_trial=True)





