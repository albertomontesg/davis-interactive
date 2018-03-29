import os
import time

from ..connector.fabric import ServerConnectionFabric
from ..utils.scribbles import fuse_scribbles, scribbles2mask

__all__ = ['DavisInteractiveSession']


class DavisInteractiveSession:
    def __init__(self,
                 host='localhost',
                 key=None,
                 connector=None,
                 davis_root=None,
                 subset='val',
                 max_time=300,
                 max_nb_interactions=None,
                 log=False,
                 progbar=False):
        # self.host = host
        # self.key = key
        self.davis_root = davis_root or os.environ.get('DAVIS_DATASET')
        if self.davis_root is None:
            raise ValueError(
                'Davis root dir not especified. Please specify it in the environmental variable DAVIS_DATASET or give it as parameter in davis_root.'
            )

        if log and progbar:
            raise ValueError('log and progbar, only one can be set to True.')

        self.subset = subset
        self.max_time = min(max_time,
                            10 * 60) if max_time is not None else max_time
        self.max_nb_interactions = min(
            max_nb_interactions,
            16) if max_nb_interactions is not None else max_nb_interactions

        self.log = log
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
        self.samples = samples

        if self.log:
            print(f'Session consist on {len(self.samples)} samples.')
        elif self.progbar:
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

    def is_running(self, log=True, progbar=False):
        if log and progbar:
            raise ValueError('log and progbar, only one can be set to True.')

        # Here start counter for this interaction, and keep track to move to
        # the next sequence and so on

        c_time = time.time()
        max_time = self.max_time()

        # sample_change = self.sample_idx < 0
        sample_change = self.sample_idx < 0
        if self.max_nb_interactions:
            sample_change |= self.interaction_nb >= self.max_nb_interactions
        if self.max_time and self.sample_start_time:
            _, _, nb_objects = self.samples[self.sample_idx]
            max_time = self.max_time * nb_objects
            sample_change |= (c_time - self.sample_start_time) > max_time

        if sample_change:
            self.sample_idx += 1
            self.sample_idx = max(self.sample_idx, 0)
            self.interaction_nb = 0
            self.sample_start_time = time.time()
            self.sample_scribbles = None
            self.sample_last_scribble = None

            if self.progbar:
                _ = self.progbar.update(1)

        if self.progbar:
            seq, _, _ = self.samples[self.sample_idx]
            self.progbar.desc = f'Evaluating {seq} ' + \
                    f'Interaction {self.interaction_nb}'

        end = self.sample_idx >= len(self.samples)
        if end and self.progbar:
            self.progbar.close()
            self.progbar = None
        return not end

    def get_scribbles(self, only_last=False):  #, return_scribbles_mask=False):
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
        # if return_scribbles_mask:
        #     scribbles = scribbles2masks(scribbles)
        return sequence, scribbles, new_sequence

    def submit_masks(self, pred_masks):
        if not self.running_model:
            raise RuntimeError(
                'You must have called .get_scribbles before submiting the masks'
            )

        time_end = time.time()
        timing = time_end - self.interaction_start_time

        self.interaction_nb += 1
        sequence, scribble_idx, _ = self.samples[self.sample_idx]

        self.sample_last_scribble = self.connector.submit_masks(
            sequence, scribble_idx, pred_masks, timing, self.interaction_nb)
        self.sample_scribbles = fuse_scribbles(self.sample_scribbles,
                                               self.sample_last_scribble)

        self.running_model = False

    def get_report(self):
        return self.connector.get_report()
