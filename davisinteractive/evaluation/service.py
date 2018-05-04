from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from .. import logging
from ..dataset.davis import Davis
from ..metrics import batched_jaccard
from ..robot import InteractiveScribblesRobot
from ..storage import LocalStorage


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

    ROBOT_DEFAULT_PARAMETERS = {
        'kernel_size': .2,
        'max_kernel_radius': 16,
        'min_nb_nodes': 4,
        'nb_points': 1000
    }

    def __init__(self,
                 subset,
                 storage=None,
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
        if subset != 'test-dev':
            logging.verbose('Checking DAVIS dataset files', 1)
            self.davis.check_files(self.sequences)

        # Init storage
        self.storage = storage or LocalStorage()

        # Parameters
        self.max_t = max_t
        self.max_i = max_i

        # Num entries
        self.num_entries = 0
        for seq in self.sequences:
            nb_scribbles = Davis.dataset[seq]['num_scribbles']
            nb_frames = Davis.dataset[seq]['num_frames']
            nb_objects = Davis.dataset[seq]['num_objects']
            self.num_entries += nb_scribbles * nb_frames * nb_objects

        if self.max_i is not None:
            self.num_entries *= self.max_i

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

    def post_predicted_masks(self, sequence, scribble_idx, pred_masks, timing,
                             interaction, user_key, session_key):
        """ Post the predicted masks and return new scribble.

        When the predicted masks are given, the metrics are computed and stored.

        # Arguments
            sequence: String. Sequence name of the predicted masks.
            scribble_idx: Integer. Scribble index of the sample evaluating.
            pred_masks: Numpy Array. Predicted masks for the given sequence.
            timing: Float. Timing in seconds of this interaction.
            interaction: Integer. Interaction number.
            user_key: String. User identifier.
            session_key: String. Session identifier.

        # Returns
            Dictionary: Scribble returned by the scribble robot

        # Raises
            RuntimeError: When a previous interaction is missing, or the
            interaction has already been submitted.
        """
        # Load ground truth masks and compute jaccard metric
        gt_masks = self.davis.load_annotations(sequence)
        nb_objects = Davis.dataset[sequence]['num_objects']

        jaccard = batched_jaccard(
            gt_masks,
            pred_masks,
            average_over_objects=False,
            nb_objects=nb_objects)
        nb_frames, _ = jaccard.shape

        frames_idx = np.arange(nb_frames)
        objects_idx = np.arange(nb_objects) + 1

        objects_idx, frames_idx = np.meshgrid(objects_idx, frames_idx)

        # Save the results on storage
        self.storage.store_interactions_results(
            user_key, session_key, sequence, scribble_idx, interaction, timing,
            objects_idx.ravel().tolist(),
            frames_idx.ravel().tolist(),
            jaccard.ravel().tolist())

        # Generate next scribble
        worst_frame = jaccard.mean(axis=1).argmin()
        next_scribble = self.robot.interact(
            sequence,
            pred_masks,
            gt_masks,
            nb_objects=nb_objects,
            frame=worst_frame)

        return next_scribble

    def get_report(self, **kwargs):
        """ Get report for a session.

        # Arguments
            user_key: String. User identifier.
            session_key: String. Session identifier.

        # Returns
            Report
        """
        return self.storage.get_report(**kwargs)
