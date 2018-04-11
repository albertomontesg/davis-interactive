from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from .. import logging
from ..dataset.davis import Davis
from ..metrics import batched_jaccard
from ..robot import InteractiveScribblesRobot


class EvaluationService:

    VALID_SUBSETS = ['train', 'val', 'test-dev']
    REPORT_COLUMNS = [
        'sequence', 'scribble_idx', 'interaction', 'object_id', 'frame',
        'jaccard', 'timming'
    ]
    ROBOT_DEFAULT_PARAMETERS = {
        'kernel_size': .2,
        'max_kernel_radius': 16,
        'min_nb_nodes': 4,
        'nb_points': 1000
    }

    def __init__(self, davis_root=None, robot_parameters=None):
        self.davis = Davis(davis_root=davis_root)
        self.session_started = False

        robot_parameters = robot_parameters or self.ROBOT_DEFAULT_PARAMETERS
        self.robot = InteractiveScribblesRobot(**robot_parameters)

        self.sequences = None
        self.sequences_scribble_idx = None
        self.report = None

    def start(self, subset):
        if subset not in self.davis.sets:
            raise ValueError('Subset must be a valid subset: {}'.format(
                self.davis.sets.keys()))

        # Get the list of sequences to evaluate and also from all the scribbles
        # available
        self.sequences = self.davis.sets[subset]
        self.sequences_scribble_idx = []
        for s in self.sequences:
            nb_scribbles = self.davis.dataset['sequences'][s]['num_scribbles']
            nb_objects = self.davis.dataset['sequences'][s]['num_objects']
            for i in range(1, nb_scribbles + 1):
                self.sequences_scribble_idx.append((s, i, nb_objects))

        # Check all the files are placed
        logging.verbose('Checking DAVIS dataset files', 1)
        self.davis.check_files(self.sequences)

        # Create empty report
        self.report = pd.DataFrame(columns=self.REPORT_COLUMNS)

        self.session_started = True
        max_t, max_i = None, None

        logging.info('Starting evaluation session')
        return self.sequences_scribble_idx, max_t, max_i

    def get_starting_scribble(self, sequence, scribble_idx):
        if not self.session_started:
            raise RuntimeError('Session not started')
        if sequence not in self.sequences:
            raise ValueError('Invalid sequence: %s' % sequence)
        if (sequence, scribble_idx) not in [
            (s, i) for s, i, _ in self.sequences_scribble_idx
        ]:
            raise ValueError('Invalid scribble index: {}'.format(scribble_idx))

        scribble = self.davis.load_scribble(sequence, scribble_idx)

        return scribble

    def submit_masks(self, sequence, scribble_idx, pred_masks, timming,
                     interaction):
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

    def close(self):
        pass
