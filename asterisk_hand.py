import csv


class HandObj:
    def __init__(self, name, fingers=None):
        '''
        Class which stores relevant hand information.
        :param hand_name - name of the hand
        :param span - max span measurement, precision grasp
        :param depth - max depth measurement, precision grasp
        :param num_fingers - number of fingers on hand
        '''
        spans, depths = self._load_measurements()

        self.hand_name = name
        # TODO: edit here when load measurements function is done
        self.span = spans[name]
        self.depth = depths[name]
        self.num_fingers = fingers
        # TODO: decide how to check if two hands are the same. Just go by name? Or check everything?

    def get_name(self):
        '''
        Getter for hand name
        '''
        return self.hand_name

    def _load_measurements(self):
        '''
        Get hand span and depth measurements from file
        '''
        # import hand span data
        spans = dict()
        depths = dict()

        #print("LOADING HAND MEASUREMENTS")
        with open(".hand_dimensions") as csv_file:
            csv_reader_hands = csv.reader(csv_file, delimiter=',')
            # populating dictionaries with dimensions
            for row in csv_reader_hands:
                # TODO: make it so that we just return the hand span and depth that we need
                hand_name = row[0]
                hand_span = float(row[1])
                hand_depth = float(row[2])

                spans[hand_name] = hand_span
                depths[hand_name] = hand_depth

        return spans, depths


h_two_v_two = HandObj("2v2", 2)
h_two_v_three = HandObj("2v3", 2)
h_three_v_three = HandObj("3v3", 2)

h_basic = HandObj("basic", 2)
h_m_stiff = HandObj("m2stiff", 2)
h_m_active = HandObj("m2active", 2)
h_m_vf = HandObj("modelvf", 2)

h_barrett = HandObj("barrett", 3)