from __future__ import absolute_import, division

import os
import random
import time
from copy import deepcopy
from datetime import datetime

from .. import logging
from ..connector.fabric import ServerConnectionFabric
from ..utils.scribbles import fuse_scribbles

__all__ = ['DavisInteractiveSession']


class DavisInteractiveSession:
    """ Class which allows to interface with the evaluation.

    # Arguments
        host: String. Host of the evuation server. Only `localhost`
            available for now.
        davis_root: String. Path to the Davis dataset root path. Necessary
            for evaluation when `host='localhost'`.
        subset: String. Subset to evaluate. If `host='localhost'` subset
            can only be `train` or `val` subsets. If the evaluation is
            performed against a remote server, this parameter is ignored
            and the evaulated subset will be given by the remote server.
        shuffle: Boolean. Shuffle the samples when evaluating.
        max_time: Integer. Number of seconds maximum to evaluate a single
            sample.
        max_nb_interactions: Integer. Maximum number of interactions to
            evaluate per sample.
        report_save_dir: String. Path to the directory where the report will
            be stored during the evaluation. By default is the current working
            directory. A temporal file will be storing snapshots of the results
            on this same directory with a suffix `.tmp`.
        progbar: Boolean. Wether to show a progbar to show the evolution
            of the evaluation. If `True`, `tqdm` Python package will be
            required.
    """

    def __init__(self,
                 host='localhost',
                 key=None,
                 connector=None,
                 davis_root=None,
                 subset='val',
                 shuffle=False,
                 max_time=None,
                 max_nb_interactions=5,
                 report_save_dir=None,
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

        self.report_save_dir = report_save_dir or os.getcwd()
        # Crete the directory if does not exists
        if not os.path.exists(self.report_save_dir):
            os.makedirs(self.report_save_dir)
        self.report_name = 'result_%s' % datetime.now().strftime(
            '%Y%m%d_%H%M%S')

    def __enter__(self):
        # Create connector
        samples, max_t, max_i = self.connector.start_session(
            self.subset, davis_root=self.davis_root)
        if self.shuffle:
            logging.verbose('Shuffling samples', 1)
            random.shuffle(samples)
        self.samples = samples

        logging.info('Started session with {} samples'.format(
            len(self.samples)))

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

    def next(self):
        """ Iterate to the next iteration/sample of the evaluation process.

        This function moves the iteration to the next iteration or to the next
        sample in case the maximum number of iterations or maximum time have
        been hit.
        This function can be used as control flow on user's code to know until
        which iteration the evuation is being performed.

        # Returns
            bool: Indicates whether the evaluation is still taking place.
        """

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
            logging.info('Start evaluation for sequence %s' % seq)

        # Save report on final version if the evaluation ends
        if end:
            df = self._get_report()
            report_filename = os.path.join(self.report_save_dir,
                                           '%s.csv' % self.report_name)
            df.to_csv(report_filename)
            # Remove the temporal file
            tmp_report_filename = os.path.join(self.report_save_dir,
                                               '%s.tmp.csv' % self.report_name)
            os.remove(tmp_report_filename)

        return not end

    # , return_scribbles_mask=False):
    def get_scribbles(self, only_last=False):
        """ Ask for the next scribble

        There is the possibility to ask for only the last scribble. By default,
        all scribbles obtained for the current sample are returned.

        This method returns information about the sequence of the sample being
        evaluated, the scribbles and whether it is a new sample. This information
        might be useful for the user to perform any operation like loading a
        model for a new sequence.

        # Arguments
            only_last: Boolean.

        # Returns
            (string, dict, bool): Returns the name of the sequence of the
                current sample, the scribbles of the current sample and a
                boolean indicating whether it is the first iteration of the given
                sample, respectively.
        """
        if self.running_model:
            raise RuntimeError(
                'You can not call get_scribbles twice without submitting the '
                'masks first')

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

    def scribbles_iterator(self, *args, **kwargs):
        """ Iterate over all the samples and iterations to evaluate.

        Instead of running a while loop with
        #DavisInteractiveSession.next and then call to
        #DavisInteractiveSession.get_scribbles, you can iterate with this
        generator:

        # Example
        ```python
        for sequence, scribble, new_sequence in sess.scribbles_iterator():
            # Predict with model
        ```

        # Arguments
            *args, **kwargs: This arguments will be passed internally to
                #DavisInteractiveSession.get_scribbles method.

        # Yields
        `(string, dict, bool)`: Yields the name of the sequence of the
                current sample, the scribbles of the current sample and a
                boolean indicating if it is the first iteration of the given
                sample, respectively.
        """
        while self.next():
            yield self.get_scribbles(*args, **kwargs)

    def submit_masks(self, pred_masks):
        """ Submit the predicted masks.

        # Arguments
            pred_masks: Numpy array with the predicted mask for
                the current sample. The array must be of `dtype=np.int` and
                of size equal to the 480p resolution of the DAVIS
                dataset.
        """
        if not self.running_model:
            raise RuntimeError('You must have called .get_scribbles before '
                               'submiting the masks')

        time_end = time.time()
        timing = time_end - self.interaction_start_time
        self.interaction_start_time = None
        logging.info(
            'The model took {:.3f} seconds to make a prediction'.format(timing))

        self.interaction_nb += 1
        sequence, scribble_idx, _ = self.samples[self.sample_idx]

        self.sample_last_scribble = self.connector.submit_masks(
            sequence, scribble_idx, pred_masks, timing, self.interaction_nb)
        self.sample_scribbles = fuse_scribbles(self.sample_scribbles,
                                               self.sample_last_scribble)

        df = self._get_report()
        tmp_report_filename = os.path.join(self.report_save_dir,
                                           '%s.tmp.csv' % self.report_name)
        df.to_csv(tmp_report_filename)

        self.running_model = False

    def _get_report(self):
        """ Gives the current report of the evaluation

        # Returns
            pd.DataFrame: Dataframe with the current evaluation results.
        """
        return self.connector.get_report()
