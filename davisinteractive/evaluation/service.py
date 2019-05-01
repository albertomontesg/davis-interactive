from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from .. import logging
from ..dataset.davis import Davis
from ..metrics import batched_f_measure, batched_jaccard
from ..robot import InteractiveScribblesRobot
from ..storage import LocalStorage

ROBOT_DEFAULT_PARAMETERS = {
    'kernel_size': .2,
    'max_kernel_radius': 16,
    'min_nb_nodes': 4,
    'nb_points': 1000
}


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
        metric_to_optimize: Enum. Metric targeting to optimize. Possible values:
            J, F or J_AND_F.
        time_threshold: Integer. Time in seconds to use it as threshold to
            compute the jaccard and compare the evaluation of different methods.
    """

    _AVAILABLE_METRICS = ('J', 'F', 'J_AND_F')

    def __init__(self,
                 subset,
                 storage=None,
                 davis_root=None,
                 robot_parameters=None,
                 max_t=None,
                 max_i=None,
                 metric_to_optimize='J_AND_F',
                 time_threshold=None):
        if subset not in Davis.sets:
            raise ValueError('Subset must be a valid subset: {}'.format(
                Davis.sets.keys()))

        self.davis = Davis(davis_root=davis_root)

        robot_parameters = robot_parameters or ROBOT_DEFAULT_PARAMETERS
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
        if metric_to_optimize not in self._AVAILABLE_METRICS:
            raise ValueError(
                'metric_to_optimize not between available metrics: {}'.format(
                    self._AVAILABLE_METRICS))
        self.metric_to_optimize = metric_to_optimize

        # Num entries
        self.num_entries = 0
        self.total_nb_objects = 0
        for seq in self.sequences:
            nb_scribbles = Davis.dataset[seq]['num_scribbles']
            nb_frames = Davis.dataset[seq]['num_frames']
            nb_objects = Davis.dataset[seq]['num_objects']
            self.num_entries += nb_scribbles * nb_frames * nb_objects
            self.total_nb_objects += nb_objects
        self.avg_nb_objects = self.total_nb_objects / len(self.sequences)
        # self.global_timeout = self.avg_nb_objects * self.max_t
        self.time_threshold = time_threshold or 60  # seconds

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

    def post_predicted_masks(self,
                             sequence,
                             scribble_idx,
                             pred_masks,
                             timing,
                             interaction,
                             user_key,
                             session_key,
                             next_scribble_frame_candidates=None):
        """ Post the predicted masks and return new scribble.

        When the predicted masks are given, the metrics are computed and
        stored.

        # Arguments
            sequence: String. Sequence name of the predicted masks.
            scribble_idx: Integer. Scribble index of the sample evaluating.
            pred_masks: Numpy Array. Predicted masks for the given sequence.
            timing: Float. Timing in seconds of this interaction.
            interaction: Integer. Interaction number.
            user_key: String. User identifier.
            session_key: String. Session identifier.
            next_scribble_frame_candidates: List of Integers. Optional value
                specifying the possible frames from which generate the next
                scribble. If values given, the next scribble will be performed
                in the frame where the evaluation metric scores the least on
                the list of given frames. Invalid frames indexes are ignored.

        # Returns
            Dictionary: Scribble returned by the scribble robot

        # Raises
            RuntimeError: When a previous interaction is missing, or the
                interaction has already been submitted.
            ValueError: When interaction is higher than the maximum number of
                interactions in the evaluation.
        """
        if self.max_i and interaction > self.max_i:
            raise ValueError(
                'Interaction {} is higher than'.format(interaction) +
                ' the maximum number of interactions {}'.format(self.max_i))
        if interaction < 1:
            raise ValueError(
                'Interaction value invalid. Should be higher than 0.')
        if (sequence, scribble_idx) not in self.sequences_scribble_idx:
            raise ValueError(
                'Sequence: {} and scribble index: {} invalid'.format(
                    sequence, scribble_idx))

        # Load ground truth masks and compute jaccard metric
        gt_masks = self.davis.load_annotations(sequence)
        nb_objects = Davis.dataset[sequence]['num_objects']

        jaccard = batched_jaccard(
            gt_masks,
            pred_masks,
            average_over_objects=False,
            nb_objects=nb_objects)
        contour = batched_f_measure(
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
            jaccard.ravel().tolist(),
            contour.ravel().tolist())

        if self.metric_to_optimize == 'J':
            metric = jaccard.mean(axis=1)
        elif self.metric_to_optimize == 'F':
            metric = contour.mean(axis=1)
        elif self.metric_to_optimize == 'J_AND_F':
            metric = .5 * jaccard + .5 * contour
            metric = metric.mean(axis=1)
        else:
            raise ValueError('Invalid metric_to_optimize: {}'.format(
                self.metric_to_optimize))

        prev_frames_used = self.storage.get_annotated_frames(
            session_key, sequence, scribble_idx)
        prev_frames_used = set(prev_frames_used)

        override = next_scribble_frame_candidates is not None
        next_scribble_frame_candidates = next_scribble_frame_candidates or [
            i for i in range(nb_frames)
        ]
        next_scribble_frame_candidates = set(next_scribble_frame_candidates)

        # Next frames candidates will be the ones requested by the used except
        # the ones previously used.
        next_frame = next_scribble_frame_candidates - prev_frames_used
        if not next_frame:
            # In case all the request frames have been previously used, use all
            # the requested frames as candidates.
            next_frame = next_scribble_frame_candidates

        metric_idx = metric.argsort()
        i = 0
        while i < nb_frames and metric_idx[i] not in next_frame:
            i += 1

        next_frame = metric_idx[i]
        self.storage.store_annotated_frame(session_key, sequence, scribble_idx,
                                           next_frame, override)

        # Generate next scribble
        next_scribble = self.robot.interact(
            sequence,
            pred_masks,
            gt_masks,
            nb_objects=nb_objects,
            frame=next_frame)

        return next_scribble

    def get_report(self, **kwargs):
        """ Get report for a session.

        # Arguments
            user_key: String. User identifier.
            session_key: String. Session identifier.

        # Returns
            Pandas DataFrame: Report.
        """
        return self.storage.get_report(**kwargs)

    def summarize_report(self, df):
        """ Given a report it will reconstruct the missing entries and compute
        a summarization of it.

        # Arguments
            df: Pandas DataFrame. The report to summarize.

        # Returns
            Dictionary: with different scores computed and the curve values
        """
        if len(df) == 0:
            df = df.set_index(
                ['interaction', 'sequence', 'scribble_idx', 'object_id'])
        else:
            df = df.groupby(
                ['interaction', 'sequence', 'scribble_idx',
                 'object_id']).mean()
        if 'frame' in df:
            df = df.drop(columns='frame')
        if 'session_id' in df:
            df = df.drop(columns='session_id')

        dfr = self._reconstruct_report(df)
        df_average = dfr.groupby(['interaction']).mean()
        df_average['timing'] = df_average['timing'].cumsum()
        df_average.loc[0] = [0, 0, 0, 0]
        df_average = df_average.sort_index()

        df_time = dfr.reset_index().groupby(
            ['sequence', 'scribble_idx', 'interaction']).agg({
                'timing': 'mean',
            })
        df_time = df_time.groupby('interaction').mean()
        df_time['timing'] = df_time['timing'].cumsum()
        df_time.loc[0] = [0]
        df_time = df_time.sort_index()

        time = df_time['timing'].values
        if self.metric_to_optimize == 'J':
            metric = df_average['jaccard'].values
        elif self.metric_to_optimize == 'F':
            metric = df_average['contour'].values
        elif self.metric_to_optimize == 'J_AND_F':
            metric = df_average['j_and_f'].values
        else:
            raise ValueError('Invalid metric_to_optimize: {}'.format(
                self.metric_to_optimize))

        if self.max_t:
            global_timeout = self.avg_nb_objects * self.max_t
        else:
            max_interactions = self.max_i or df.reset_index(
            )['interaction'].max()
            global_timeout = max_interactions * df['timing'].max()

        time = np.concatenate((time, [global_timeout]))
        metric = np.concatenate((metric, metric[-1:]))

        metric_th = np.interp(self.time_threshold, time, metric)
        if time.max() == 0.:
            auc = 0.
        else:
            auc = np.trapz(metric, x=time) / time.max()

        return {
            'auc': auc,
            'metric_at_threshold': {
                'threshold': self.time_threshold,
                self.metric_to_optimize: metric_th
            },
            'curve': {
                'time': time.tolist(),
                self.metric_to_optimize: metric.tolist()
            }
        }

    def _reconstruct_report(self, df):
        """ Reconstruct the report with missing entries.

        In case the timeout has been reached and some interactions have not been
        evaluated, the reconstruction ensure to have a result for every
        interaction putting the jaccard of the previous evaluated interaction
        with a timing cost of 0.
        """
        index = []
        max_i = self.max_i or df.reset_index()['interaction'].max()
        if np.isnan(max_i):
            max_i = 1
        for i in range(max_i):
            for seq in self.sequences:
                nb_scribbles = Davis.dataset[seq]['num_scribbles']
                nb_objects = Davis.dataset[seq]['num_objects']
                for j in range(nb_scribbles):
                    for k in range(nb_objects):
                        index.append((i + 1, seq, j + 1, k + 1))

        index = pd.MultiIndex.from_tuples(
            index,
            names=['interaction', 'sequence', 'scribble_idx', 'object_id'])
        df = df.reindex(index)

        for seq in self.sequences:
            nb_scribbles = Davis.dataset[seq]['num_scribbles']
            nb_objects = Davis.dataset[seq]['num_objects']

            for scribble_idx in range(1, nb_scribbles + 1):
                prev_result = np.zeros((nb_objects, 4), dtype=np.float)
                for it in range(1, max_i + 1):
                    result_iter = df.loc[it, seq, scribble_idx, :]
                    if np.any(pd.isna(result_iter)):
                        prev_result[:, -1] = 0
                        df.loc[it, seq, scribble_idx, :] = prev_result
                    else:
                        prev_result = result_iter.values
        return df
