from __future__ import absolute_import, division

import hashlib
import json
import os
import tempfile
import urllib.request as request
import zipfile

import numpy as np
from PIL import Image

from .. import Path, logging


class Davis:
    """ DAVIS class to encapsulate some information about the dataset.

    This class only needs to have specified the root path. When done, some
    atributes can be accessible like the sequence list for every subset, or
    specific information for every sequence like the number of frames, the
    number of objects or the image size for every sequence.
    For more information about the sequence attributes available, check this
    [file](https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/dataset/davis.json).

    # Arguments
        davis_root: String. Path to the DAVIS dataset files. This argument can
            be left as `None` and specify it as an environtmental variable
            `DATASET_DAVIS`. This usage is useful in the case a group of
            people is working with the same code and every one has a different
            path where the DAVIS dataset is stored.

    # Attributes
        ANNOTATIONS_SUBDIR: Relative path with respect to the root path where
            the ground truth masks are stored. (Annotations)
        SCRIBBLES_SUBDIR: Relative path with respect to the root path where the
            scribbles are stored. (Scribbles)
        RESOLUTION: Resolution of the dataset used to perform all the
            evaluation. (480p)
        sets: Dictionary. The keys are all the DAVIS dataset subsets and the
            values are the list of sequences belonging to that subset.
        dataset: Dictionary. Contains all the information from all the dataset.

    # Raises
        ValueError: if neither `davis_root` or environmental variable
            `DATASET_DAVIS` are specified.
    """

    # Download information
    # pylint: disable=line-too-long
    SCRIBBLES_URL = 'https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip'
    SCRIBBLES_HASH = '6c6811c67ef757091212a98b68b841305f92b57f6cd2938e0fa94ae8591c3226'
    # pylint: enable=line-too-long

    ANNOTATIONS_SUBDIR = "Annotations"
    SCRIBBLES_SUBDIR = "Scribbles"
    RESOLUTION = "480p"

    def __init__(self, davis_root=None):
        self.davis_root = davis_root or os.environ.get('DATASET_DAVIS')
        self.davis_root = Path(self.davis_root)
        if not self.davis_root.exists() or not self.davis_root.is_dir():
            raise ValueError(
                'Davis root dir not especified. Please specify it in the '
                'environmental variable DAVIS_DATASET or give it as parameter '
                'in davis_root.')

        # Load DAVIS data
        with Path(__file__).parent.joinpath('davis.json').open() as fp:
            self.dataset = json.load(fp)
        logging.verbose('Loaded dataset data', 2)

        self.sets = {s: [] for s in self.dataset['sets']}
        logging.verbose('Extracted sequences for each subset', 2)

        for s in self.dataset['sequences'].values():
            self.sets[s['set']].append(s['name'])

    def _download_scribbles(self):
        file_name = Path(self.SCRIBBLES_URL).name
        download_file = Path(tempfile.mkdtemp()) / file_name

        # Downloading
        logging.info('Downloading Scribbles')
        request.urlretrieve(self.SCRIBBLES_URL, download_file)

        # Check integrity
        logging.info('Checking hash')
        md5 = hashlib.sha256(download_file.read_bytes()).hexdigest()
        if md5 != self.SCRIBBLES_HASH:
            raise ValueError(('Downloaded file do not have a correct hash.\n'
                              'Expected {}\n'
                              'Obtained: {}').format(self.SCRIBBLES_HASH, md5))

        # Extract file
        logging.info('Extracting file')
        zipfile.ZipFile(download_file).extractall(self.davis_root.parent)

        logging.info('Download completed')

    def _check_annotations_files(self, sequences):
        pass

    def _check_scribbles_files(self, sequences):
        pass

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
                        os.path.join(seq_scribbles_path,
                                     '{:03d}.json'.format(i))):
                    self._download_scribbles()
                    if not os.path.exists(
                            os.path.join(seq_scribbles_path,
                                         '{:03d}.json'.format(i))):
                        raise FileNotFoundError(
                            ('Scribble file not found for sequence '
                             '{} and scribble {}').format(seq, i))

            # Check annotations files required for the evaluation
            nb_frames = self.dataset['sequences'][seq]['num_frames']
            for i in range(nb_frames):
                if not os.path.exists(
                        os.path.join(seq_annotations_path,
                                     '{:05d}.png'.format(i))):
                    raise FileNotFoundError(('Annotations file not found for '
                                             'sequence {} and frame {}').format(
                                                 seq, i))

    def load_scribble(self, sequence, scribble_idx):
        scribble_file = os.path.join(self.davis_root, Davis.SCRIBBLES_SUBDIR,
                                     sequence,
                                     '{:03d}.json'.format(scribble_idx))

        with open(scribble_file, 'r') as fp:
            scribble_data = json.load(fp)
        assert scribble_data['sequence'] == sequence

        logging.verbose(
            'Loaded scribble for sequence {} and scribble_idx {}'.format(
                sequence, scribble_idx), 1)
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
            mask = Image.open(os.path.join(root_path, '{:05d}.png'.format(f)))
            mask = np.asarray(mask)
            assert mask.shape == tuple(img_size[::-1])
            annotations[f] = mask

        logging.verbose('Loaded annotations for sequence %s' % sequence, 1)
        logging.verbose('at path: %s' % root_path, 2)
        logging.verbose('Annotations shape: {}'.format(annotations.shape), 2)

        return annotations
