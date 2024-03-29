"""
Class which stores the locations of files.
"""


from pathlib import Path


class AstDirectory:

    def __init__(self, home_directory):
        # TODO: maybe save the file names as entries in a dictionary? Revisit.
        self.data_home = home_directory
        self.compressed_data = None
        self.aruco_pics = None
        self.aruco_data = None
        self.path_data = None
        self.metric_results = None
        self.result_figs = None
        self.debug_figs = None
        self.resources = None

    # def file_location(self):
    #     """
    #     Function enables user to edit a file location. Function handles pathlib stuff
    #     """
    #     pass

    def import_locations(self, file_loc_key, file_loc):
        """
        Function imports a csv file and updates the object with locations
        """
        dict_keys = ["compressed_data", "aruco_pics", "aruco_data", "path_data", "metric_results", "result_figs", "debug_figs"]
        
        pass

    # def is_valid(self):
    #     """
    #     Function returns bool if all of the required file locations are added.
    #     """
    #     pass

# no __name__ == __main__ here, want this to be accessible
home_directory = Path(__file__).parent.parent.absolute()
data_directory = home_directory / "data"

# put together an AstDirectory manually
my_ast_files = AstDirectory(data_directory)
#my_ast_files.data_home = data_directory
my_ast_files.compressed_data = data_directory / "compressed_data"
my_ast_files.aruco_pics = data_directory / "viz"
my_ast_files.aruco_data = data_directory / "aruco_data"
my_ast_files.path_data = data_directory / "trial_paths"
my_ast_files.metric_results = data_directory / "results"
my_ast_files.result_figs = data_directory / "results" / "plots"
my_ast_files.debug_figs = data_directory / "results" / "debug_plots"
my_ast_files.resources = data_directory.parent / "resources"


new_home_directory = Path("/home/john/Programs/new_ast_data")
data_directory = new_home_directory
new_ast_files = AstDirectory(new_home_directory)
#new_ast_files.data_home = data_directory
new_ast_files.compressed_data = data_directory / "compressed_data"
new_ast_files.aruco_pics = data_directory / "viz"
new_ast_files.aruco_data = data_directory / "aruco_data"
new_ast_files.path_data = data_directory / "trial_paths"
new_ast_files.metric_results = data_directory / "results"
new_ast_files.result_figs = data_directory / "results" / "plots"
new_ast_files.debug_figs = data_directory / "results" / "debug_plots"

resources_home = Path(__file__).parent.parent.absolute()
new_ast_files.resources = resources_home.parent / "resources"