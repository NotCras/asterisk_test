"""
An object which holds important information about the hand, readily accessible when needed for normalizing translation
Holds a hand's name, span & depth measurements, number of fingers.
Loads this information from hidden file in the root.
"""
import pdb

import pandas as pd
from numpy import abs
from file_manager import my_ast_files
from pathlib import Path


class HandInfo:
    def __init__(self, name, num_fingers=None):
        """
        Class which stores relevant hand information.
        :param hand_name: name of the hand
        :param num_fingers: number of fingers on hand
        """
        self.hand_name = name
        self.span, self.depth, self.aruco_id = self._load_measurements()
        self.num_fingers = num_fingers

    def get_name(self):
        """
        Getter for hand name
        """
        return self.hand_name

    def get_span(self):
        """
        Getter for span val (mm)
        """
        return self.span

    def get_depth(self):
        """
        Getter for depth val (mm)
        """
        return self.depth

    def compare_hands(self, other_hand_info, tolerance=5):
        """
        Compares two hand info objects to each other. First checks name, then span, then depth.
        Can also set a tolerance for how to compare spans and depths with
        Returns T/F
        """
        if self.get_name() == other_hand_info.get_name():
            if abs(self.span - other_hand_info.span) <= tolerance:
                if abs(self.depth - other_hand_info.depth) <= tolerance:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def _load_measurements(self):
        """
        Get hand span and depth measurements from file
        """
        home_directory = Path(__file__).parent.absolute()
        hand_stats_loc = str(home_directory) + '/.hand_stats'
        dims_df = pd.read_csv(hand_stats_loc,
                              index_col=0)  # names=['name', 'mx_span', 'mx_depth', 'id_num'], index_col=0)
        dims = dims_df.loc[self.hand_name]
        span = dims[0]  # mx_span
        depth = dims[1]  # mx_depth
        aruco_id = dims[2]  # .id_num

        return span, depth, aruco_id

def get_hand_stats(specific_hand = None):
    """
    Gets the data out of the hand_stats file
    """
    home_directory = Path(__file__).parent.absolute()
    hand_stats_loc = str(home_directory) + '/.hand_stats'
    dims_df = pd.read_csv(hand_stats_loc, index_col=0) #names=['name', 'mx_span', 'mx_depth', 'id_num'], index_col=0)

    if specific_hand is None:
        return dims_df
    else:
        # returning piecemeal so I don't have to rely on pandas row obj
        dims = dims_df.loc[specific_hand]
        span = dims[0]  # mx_span
        depth = dims[1]  # mx_depth
        aruco_id = dims[2]  # .id_num
        return span, depth, aruco_id


if __name__ == '__main__':
    h_two_v_two = HandInfo("2v2", 2)
    h_two_v_three = HandInfo("2v3", 2)
    h_three_v_three = HandInfo("3v3", 2)

    h_basic = HandInfo("basic", 2)
    h_m_stiff = HandInfo("m2stiff", 2)
    h_m_active = HandInfo("m2active", 2)
    h_m_vf = HandInfo("modelvf", 2)

    h_barrett = HandInfo("barrett", 3)