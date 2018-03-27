import json
import os

import numpy as np
from PIL import Image


class DAVIS:

    ANNOTATIONS_DIR = 'Annotations'
    SCRIBBLES_DIR = 'Scribbles'
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

        self.sets = {s: [] for s in self.dataset['sets']}

        for s in self.dataset['sequences'].values():
            self.sets[s['set']].append(s['name'])

    def check_files(self, sequences):
        for seq in sequences:
            seq_scribbles_path = os.path.join(self.davis_root,
                                              DAVIS.SCRIBBLES_DIR, seq)
            seq_annotations_path = os.path.join(
                self.davis_root, DAVIS.ANNOTATIONS_DIR, DAVIS.RESOLUTION, seq)

            # Check scribbles files needed to give them as base for the user
            nb_scribbles = self.dataset['sequences'][seq]['num_scribbles']
            for i in range(1, nb_scribbles):
                if not os.path.exists(
                        os.path.join(seq_scribbles_path, f'{i:03d}.json')):
                    raise FileNotFoundError(
                        f'Scribble file not found for sequence {seq} and scribble {i}'
                    )

            # Check annotations files required for the evaluation
            nb_frames = self.dataset['sequences'][seq]['num_frames']
            for i in nb_frames:
                if not os.path.exists(
                        os.path.join(seq_annotations_path, f'{i:05d}.png')):
                    raise FileNotFoundError(
                        f'Annotations file not found for sequence {seq} and frame {i}'
                    )

    def load_scribble(self, sequence, scribble_idx):
        scribble_file = os.path.join(self.davis_root, DAVIS.SCRIBBLES_DIR,
                                     sequence, f'{scribble_idx:03d}.json')

        with open(scribble_file, 'r') as fp:
            scribble_data = json.load(fp)
        assert scribble_data['sequence'] == sequence

        return scribble_data

    def load_annotations(self, sequence):
        root_path = os.path.join(self.davis_root, DAVIS.ANNOTATIONS_DIR,
                                 DAVIS.RESOLUTION, sequence)
        num_frames = self.dataset['sequences'][sequence]['num_frames']
        img_size = self.dataset['sequence'][sequence]['image_size']

        annotations = np.empty(
            (num_frames, img_size[1], img_size[0]), dtype=np.int)

        for f in range(num_frames):
            mask = Image.open(os.path.join(root_path, f'{f:05d}.png'))
            mask = np.asarray(mask)
            assert mask.shape == tuple(img_size[::-1])
            annotations[f] = mask

        return annotations
