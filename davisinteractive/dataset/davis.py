import json
import os

import numpy as np
from PIL import Image

from .. import logging


class Davis:
    """ DAVIS class to encapsulate some information about the dataset.

    """

    ANNOTATIONS_SUBDIR = 'Annotations'
    SCRIBBLES_SUBDIR = 'Scribbles'
    RESOLUTION = '480p'

    def __init__(self, davis_root=None):
        self.davis_root = davis_root or os.environ.get('DATASET_DAVIS')
        if self.davis_root is None:
            raise ValueError(
                'Davis root dir not especified. Please specify it in the ' +
                'environmental variable DAVIS_DATASET or give it as parameter '
                + 'in davis_root.')

        # Load DAVIS data
        with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'davis.json'),
                'r') as fp:
            self.dataset = json.load(fp)
        logging.verbose('Loaded dataset data', 2)

        self.sets = {s: [] for s in self.dataset['sets']}
        logging.verbose('Extracted sequences for each subset', 2)

        for s in self.dataset['sequences'].values():
            self.sets[s['set']].append(s['name'])

    def check_files(self, sequences):
        for seq in sequences:
            seq_scribbles_path = os.path.join(self.davis_root,
                                              Davis.SCRIBBLES_SUBDIR, seq)
            seq_annotations_path = os.path.join(self.davis_root,
                                                Davis.ANNOTATIONS_SUBDIR,
                                                Davis.RESOLUTION, seq)

            # Check scribbles files needed to give them as base for the user
            nb_scribbles = self.dataset['sequences'][seq]['num_scribbles']
            for i in range(1, nb_scribbles):
                if not os.path.exists(
                        os.path.join(seq_scribbles_path, f'{i:03d}.json')):
                    raise FileNotFoundError(
                        f'Scribble file not found for sequence {seq} ' +
                        f'and scribble {i}')

            # Check annotations files required for the evaluation
            nb_frames = self.dataset['sequences'][seq]['num_frames']
            for i in range(nb_frames):
                if not os.path.exists(
                        os.path.join(seq_annotations_path, f'{i:05d}.png')):
                    raise FileNotFoundError(
                        f'Annotations file not found for sequence {seq} ' +
                        f'and frame {i}')

    def load_scribble(self, sequence, scribble_idx):
        scribble_file = os.path.join(self.davis_root, Davis.SCRIBBLES_SUBDIR,
                                     sequence, f'{scribble_idx:03d}.json')

        with open(scribble_file, 'r') as fp:
            scribble_data = json.load(fp)
        assert scribble_data['sequence'] == sequence

        logging.verbose(f'Loaded scribble for sequence {sequence} and ' +
                        f'scribble_idx {scribble_idx}', 1)
        logging.verbose(scribble_file, 2)

        return scribble_data

    def load_annotations(self, sequence):
        root_path = os.path.join(self.davis_root, Davis.ANNOTATIONS_SUBDIR,
                                 Davis.RESOLUTION, sequence)
        num_frames = self.dataset['sequences'][sequence]['num_frames']
        img_size = self.dataset['sequences'][sequence]['image_size']

        annotations = np.empty(
            (num_frames, img_size[1], img_size[0]), dtype=np.int)

        for f in range(num_frames):
            mask = Image.open(os.path.join(root_path, f'{f:05d}.png'))
            mask = np.asarray(mask)
            assert mask.shape == tuple(img_size[::-1])
            annotations[f] = mask

        logging.verbose(f'Loaded annotations for sequence {sequence}', 1)
        logging.verbose(f'at path: {root_path}', 2)
        logging.verbose(f'Annotations shape: {annotations.shape}', 2)

        return annotations
