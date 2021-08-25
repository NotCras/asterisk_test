from ast_trial import AstBasicData


class AstTrialRotation(AstBasicData):
    """
    Class which handles cw/ccw trials
    """

    def __init__(self, data, rotation_label):
        self.poses = data  # TODO: placeholder for later

        self.trial_translation = 'n'
        self.trial_rotation = rotation_label

        # target path is (0,0) and target rotation should be max rotation value
        # TODO: will need to overwrite the generate target line and rotation methods

        # TODO: assess path labels will have to change

        # TODO: plot trial needs to be filled in... and check plot_orientations

        # TODO: will need to revisit how metrics are calculated

