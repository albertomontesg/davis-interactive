import os
import random
import time
from copy import deepcopy

from .. import logging
from ..connector.fabric import ServerConnectionFabric
from ..utils.scribbles import fuse_scribbles

__all__ = ['DavisInteractiveSession']


class DavisInteractiveSession:

    def __init__(self,
                 host='localhost',
                 key=None,
                 connector=None,
                 davis_root=None,
                 subset='val',
                 shuffle=False,
                 max_time=None,
                 max_nb_interactions=5,
                 progbar=False):
        # self.host = host
        # self.key = key
        self.davis_root = davis_root

        self.subset = subset
        self.shuffle = shuffle
        self.max_time = min(max_time,
                            10 * 60) if max_time is not None else max_time
        self.max_nb_interactions = min(
            max_nb_interactions,
            16) if max_nb_interactions is not None else max_nb_interactions

        self.progbar = progbar
        self.running_model = False

        self.connector = connector or ServerConnectionFabric.get_connector(
            host, key)

        self.samples = None
        self.sample_idx = None
        self.interaction_nb = None
        self.sample_start_time = None
        self.sample_scribbles = None
        self.sample_last_scribble = None
        self.interaction_start_time = None

    def __enter__(self):
        # Create connector
        # self.connector = ServerConnectionFabric.get_connector(
        #     self.host, self.key)
        samples, max_t, max_i = self.connector.start_session(
            self.subset, davis_root=self.davis_root)
        if self.shuffle:
            logging.verbose('Shuffling samples', 1)
            random.shuffle(samples)
        self.samples = samples

        logging.info(f'Started session with {len(self.samples)} samples')

        if self.progbar:
            from tqdm import tqdm
            self.progbar = tqdm(self.samples, desc='Evaluating')

        self.max_time = max_t or self.max_time
        self.max_nb_interactions = max_i or self.max_nb_interactions
        if self.max_time is None and self.max_nb_interactions is None:
            raise ValueError(
                'Both max_time and max_nb_interactions can not be None')

        self.sample_idx = -1
        self.interaction_nb = -1
        return self

    def __exit__(self, type_, value, traceback):
        self.connector.close()

    def is_running(self):

        # Here start counter for this interaction, and keep track to move to
        # the next sequence and so on

        c_time = time.time()

        # sample_change = self.sample_idx < 0
        sample_change = self.sample_idx < 0
        if self.max_nb_interactions:
            change_because_interaction = self.interaction_nb >= self.max_nb_interactions
            sample_change |= change_because_interaction
            if change_because_interaction:
                logging.info('Maximum number of interaction have been reached.')
        if self.max_time and self.sample_start_time:
            _, _, nb_objects = self.samples[self.sample_idx]
            max_time = self.max_time * nb_objects
            change_because_timing = (c_time - self.sample_start_time) > max_time
            sample_change |= change_because_timing
            if change_because_timing:
                logging.info('Maximum time per sample has been reached.')

        if sample_change:
            self.sample_idx += 1
            self.sample_idx = max(self.sample_idx, 0)
            self.interaction_nb = 0
            self.sample_start_time = time.time()
            self.sample_scribbles = None
            self.sample_last_scribble = None

        end = self.sample_idx >= len(self.samples)
        if not end and sample_change:
            seq, _, _ = self.samples[self.sample_idx]
            logging.info(f'Start evaluation for sequence {seq}')

        return not end

    # , return_scribbles_mask=False):
    def get_scribbles(self, only_last=False):
        if self.running_model:
            raise RuntimeError(
                'You can not call get_scribbles twice without submitting the masks first'
            )

        sequence, scribble_idx, _ = self.samples[self.sample_idx]
        new_sequence = False
        if self.interaction_nb == 0 and self.sample_scribbles is None:
            self.sample_scribbles = self.connector.get_starting_scribble(
                sequence, scribble_idx)
            self.sample_last_scribble = self.sample_scribbles
            new_sequence = True

        self.interaction_start_time = time.time()
        self.running_model = True

        if only_last:
            scribbles = self.sample_last_scribble
        else:
            scribbles = self.sample_scribbles

        # Create a copy to not pass a reference
        scribbles = deepcopy(scribbles)

        logging.info('Giving scribble to the user')

        return sequence, scribbles, new_sequence

    def submit_masks(self, pred_masks):
        if not self.running_model:
            raise RuntimeError(
                'You must have called .get_scribbles before submiting the masks'
            )

        time_end = time.time()
        timing = time_end - self.interaction_start_time
        logging.info(
            f'The model took {timing:.3f} seconds to make a prediction')

        self.interaction_nb += 1
        sequence, scribble_idx, _ = self.samples[self.sample_idx]

        self.sample_last_scribble = self.connector.submit_masks(
            sequence, scribble_idx, pred_masks, timing, self.interaction_nb)
        self.sample_scribbles = fuse_scribbles(self.sample_scribbles,
                                               self.sample_last_scribble)

        self.running_model = False

    def get_report(self):
        return self.connector.get_report()
