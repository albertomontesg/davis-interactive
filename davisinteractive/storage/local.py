from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from .. import logging
from .abstract import AbstractStorage


class LocalStorage(AbstractStorage):
    """ Local storage of the results.

    This class encapsulates the storage of the results into a pandas DataFrame.
    """

    def __init__(self):
        self.report = pd.DataFrame(columns=self.COLUMNS)
        logging.verbose('Report DataFrame created')

    def store_interactions_results(self, user_id, session_id, sequence,
                                   scribble_idx, interaction, timing,
                                   objects_idx, frames, jaccard):
        """ The information of a single interaction is given and stored.

        # Arguments
            user_id: String. User identifier. As it is in local, this value is
                ignored.
            session_id: String. Session identifier.
            sequence: String. Sequence name.
            scribble_idx: Integer. Scribble index of the sample.
            interaction: Integer. Interaction number.
            timing: Float. Timing in seconds that lasted the interaction.
            objects_idx: List of Integers. List of the objects identifiers that
                match with the jaccard metric.
            frames: List of Integers: List of frame index matching with the
                jaccard metric.
            jaccard: List of Floats: List of jaccard metric.
        """
        # Check the data.
        objects_idx = list(objects_idx)
        frames = list(frames)
        jaccard = list(jaccard)
        if (min(jaccard) < 0.) or (max(jaccard) > 1.):
            raise ValueError('Jaccard values must be between 0 and 1')

        nb = len(jaccard)
        if len(objects_idx) != nb or len(frames) != nb:
            raise ValueError('`jaccard`, `frames` and `objects_idx` must '
                             'have the same length')

        # Check previous entries
        if self.report.loc[(self.report.sequence == sequence) &
                           (self.report.scribble_idx == scribble_idx) &
                           (self.report.interaction == interaction)].size > 0:
            raise RuntimeError(('For {} and scribble {} already exist a '
                                'result for interaction {}').format(
                                    sequence, scribble_idx, interaction))
        if interaction > 1 and self.report.loc[(
                self.report.sequence == sequence) & (
                    self.report.scribble_idx == scribble_idx) & (
                        self.report.interaction == interaction - 1)].size == 0:
            raise RuntimeError(('For {} and scribble {} does not exist a '
                                'result for previous interaction {}').format(
                                    sequence, scribble_idx, interaction - 1))

        sample = {}
        sample['session_id'] = [session_id] * nb
        sample['sequence'] = [sequence] * nb
        sample['scribble_idx'] = [scribble_idx] * nb
        sample['interaction'] = [interaction] * nb
        sample['timing'] = [timing] * nb
        sample['object_id'] = objects_idx
        sample['frame'] = frames
        sample['jaccard'] = jaccard
        sample = pd.DataFrame(data=sample, columns=self.COLUMNS)
        self.report = pd.concat([self.report, sample], ignore_index=True)
        logging.info('Successfully stored sample interaction entry')

        return True

    def get_report(self, session_id=None, **kwargs):
        """ Return current report.

        # Arguments
            session_id: String. Session identifier

        # Returns
            Pandas DataFrame. Report in the form of the DataFrame.
        """
        df = self.report
        df = df.loc[df['session_id'] == session_id].reset_index(drop=True)
        return df
