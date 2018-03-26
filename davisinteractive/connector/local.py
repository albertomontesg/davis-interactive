import json
import os
import random
from glob import glob

from ..robot import InteractiveScribblesRobot
from .abstract import AbstractConnector


class LocalConnector(AbstractConnector):

    VALID_SUBSETS = ['train', 'val']
    RESOLUTION = '480p'
    SCRIBBLES_DIR = 'Scribbles'

    def __init__(self):
        self.davis_root = None
        self.robot = InteractiveScribblesRobot()
        self.session_started = False

    def start_session(self, subset, davis_root=None):
        self.davis_root = davis_root or os.environ.get('DATASET_DAVIS')
        if self.davis_root is None:
            raise ValueError(
                'Davis root dir not especified. Please specify it in the ' +
                'environmental variable DAVIS_DATASET or give it as parameter '
                + 'in davis_root.')

        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f'Subset must be a valid subset: {self.VALID_SUBSETS}')

        # Check all the files are placed

        # Get the list of sequences to evaluate and also from all the scribbles
        # available
        sequences = None

        self.session_started = True
        max_t, max_i = None, None
        return sequences, max_t, max_i

    def get_starting_scribble(self, sequence):
        if not self.session_started:
            raise RuntimeError('Session not started')
        scribbles_files = glob(
            os.path.join(self.davis_root, self.SCRIBBLES_DIR, self.RESOLUTION,
                         f'{sequence}*.json'))

        if not scribbles_files:
            raise RuntimeError('Scribbles not found')

        scribble_file = random.choice(scribbles_files)
        with open(scribble_file, 'r') as fp:
            scribble = json.load(fp)

        return scribble

    def submit_masks(self, pred_masks):
        raise NotImplementedError('This is an abstract class')

    def get_report(self):
        raise NotImplementedError('This is an abstract class')
