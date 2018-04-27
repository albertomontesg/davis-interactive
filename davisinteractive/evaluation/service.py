from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from .. import logging
from ..dataset.davis import Davis
from ..metrics import batched_jaccard
from ..robot import InteractiveScribblesRobot


class EvaluationService:
    """ Class responsible of the evaluation.

    This class is responsible of giving the samples to run the evaluation,
    to give the asked scribbles and to evaluate the interaction with the robot
    once the masks are submitted.

    # Arguments
        subset: String. Subset to evaluate. Possible values are `train`, `val`,
            'trainval' and `test-dev`.
        davis_root: String or Path. Path to the DAVIS dataset root directory,
            where the scribbles and the masks are stored.
        robot_parameters: Dictionary. Dictionary of parameters to initialize
            the scribbles robot.
        max_t: Integer. Number of seconds maximum to evaluate a single sample.
            This value will overwrite the specified from the user at
            `DavisInteractiveSession` class.
        max_i: Integer. Maximum number of interactions to evaluate per sample.
            This value will overwrite the specified from the user at
            `DavisInteractiveSession` class.
    """

    VALID_SUBSETS = ['train', 'val', 'trainval', 'test-dev']
    REPORT_COLUMNS = [
        'session_id', 'sequence', 'scribble_idx', 'interaction', 'object_id',
        'frame', 'jaccard', 'timming'
    ]
    ROBOT_DEFAULT_PARAMETERS = {
        'kernel_size': .2,
        'max_kernel_radius': 16,
        'min_nb_nodes': 4,
        'nb_points': 1000
    }

    def __init__(self,
                 subset,
                 davis_root=None,
                 robot_parameters=None,
                 max_t=None,
                 max_i=None):
        if subset not in Davis.sets:
            raise ValueError('Subset must be a valid subset: {}'.format(
                Davis.sets.keys()))

        self.davis = Davis(davis_root=davis_root)

        robot_parameters = robot_parameters or self.ROBOT_DEFAULT_PARAMETERS
        self.robot = InteractiveScribblesRobot(**robot_parameters)

        # Get the list of sequences to evaluate and also from all the scribbles
        # available
        self.sequences = self.davis.sets[subset]
        self.sequences_scribble_idx = []
        for s in self.sequences:
            nb_scribbles = Davis.dataset[s]['num_scribbles']
            for i in range(1, nb_scribbles + 1):
                self.sequences_scribble_idx.append((s, i))

        # Check all the files are placed
        logging.verbose('Checking DAVIS dataset files', 1)
        self.davis.check_files(self.sequences)

        # Create empty report
        self.report = pd.DataFrame(columns=self.REPORT_COLUMNS)

        # Parameters
        self.max_t = max_t
        self.max_i = max_i

    def get_samples(self):
        """ Get the list of samples.

        # Returns
            List of Tuples: List of pairs where the first element is the
                sequence name and the second is the scribble index to evaluate.
        """

        logging.info('Getting samples')
        return self.sequences_scribble_idx, self.max_t, self.max_i

    def get_scribble(self, sequence, scribble_idx):
        """ Get a scribble.

        # Arguments
            sequence: String. Sequence name of the scribble.
            scribble_idx: Integer. Index of the scribble to get.

        # Raises
            ValueError: when the sequence is invalid or the scribble index is
                out of range.
        """
        if sequence not in self.sequences:
            raise ValueError('Invalid sequence: %s' % sequence)
        if (sequence, scribble_idx) not in self.sequences_scribble_idx:
            raise ValueError('Invalid scribble index: {}'.format(scribble_idx))

        scribble = self.davis.load_scribble(sequence, scribble_idx)

        return scribble

    def post_predicted_masks(self, sequence, scribble_idx, pred_masks, timming,
                             interaction, user_key, session_key):
        # Evaluate the submitted masks
        if len(self.report.loc[(self.report.sequence == sequence) &
                               (self.report.scribble_idx == scribble_idx) &
                               (self.report.interaction == interaction)]) > 0:
            raise RuntimeError(
                'For {} and scribble {} already exist a result for interaction {}'.
                format(sequence, scribble_idx, interaction))
        if interaction > 1 and len(
                self.report.loc[(self.report.sequence == sequence) &
                                (self.report.scribble_idx == scribble_idx) &
                                (self.report.interaction == interaction -
                                 1)]) == 0:
            raise RuntimeError(
                'For {} and scribble {} does not exist a result for previous interaction {}'.
                format(sequence, scribble_idx, interaction - 1))
        gt_masks = self.davis.load_annotations(sequence)
        jaccard = batched_jaccard(
            gt_masks, pred_masks, average_over_objects=False)
        nb_frames, nb_objects = jaccard.shape
        nb = nb_frames * nb_objects

        frames_idx = np.arange(nb_frames)
        objects_idx = np.arange(nb_objects)

        objects_idx, frames_idx = np.meshgrid(objects_idx, frames_idx)

        uploaded_result = {k: None for k in self.REPORT_COLUMNS}
        uploaded_result['session_id'] = [session_key] * nb
        uploaded_result['sequence'] = [sequence] * nb
        uploaded_result['scribble_idx'] = [scribble_idx] * nb
        uploaded_result['interaction'] = [interaction] * nb
        uploaded_result['timming'] = [timming] * nb
        uploaded_result['object_id'] = objects_idx.ravel().tolist()
        uploaded_result['frame'] = frames_idx.ravel().tolist()
        uploaded_result['jaccard'] = jaccard.ravel().tolist()
        uploaded_result = pd.DataFrame(
            data=uploaded_result, columns=self.REPORT_COLUMNS)
        self.report = pd.concat(
            [self.report, uploaded_result], ignore_index=True)

        # Generate next scribble
        next_scribble = self.robot.interact(sequence, pred_masks, gt_masks)

        return next_scribble

    def get_report(self):
        return self.report
