"""
Class which stores the locations of files.
"""


from pathlib import Path


class AstDirectory:

    def __init__(self): 
        # TODO: maybe save the file names as entries in a dictionary? Revisit.
        self.compressed_data = None
        self.aruco_pics = None
        self.aruco_data = None
        self.path_data = None
        self.metric_results = None
        self.result_figs = None
        self.debug_figs = None

    # def file_location(self):
    #     """
    #     Function enables user to edit a file location. Function handles pathlib stuff
    #     """
    #     pass

    def import_locations(self, file_loc):
        """
        Function imports a csv file and updates the object with locations
        """
        pass

    # def is_valid(self):
    #     """
    #     Function returns bool if all of the required file locations are added.
    #     """
    #     pass