from __future__ import absolute_import, division

import hashlib
import json
import os
import tempfile
import zipfile

import numpy as np
from PIL import Image
from six.moves import urllib

from .. import logging
from ..common import Path

# Python2/3 Compatibility
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError  # pylint: disable=redefined-builtin

with Path(__file__).parent.joinpath('davis.json').open() as fp:
    _DATASET = json.load(fp)

_SETS = {s: [] for s in _DATASET['sets']}
for s in _DATASET['sequences'].values():
    _SETS[s['set']].append(s['name'])
_SETS['trainval'] = _SETS['train'] + _SETS['val']


class Davis:
    """ DAVIS class to encapsulate some information about the dataset.

    This class only needs to have the root path specified. Some
    atributes can be accessible like the sequence list for every subset, or
    specific information for every sequence like the number of frames, the
    number of objects or the image size for every sequence.
    For more information about the sequence attributes available, check this
    [file](https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/dataset/davis.json).

    # Arguments
        davis_root: String. Path to the DAVIS dataset. This argument can
            be left as `None` and specify it as an environtmental variable
            `DATASET_DAVIS`. This usage is useful in the case a group of
            people is working with the same code and every one has a different
            path where the DAVIS dataset is stored. The folder name where all
            DAVIS dataset is stored must be names `DAVIS`.

    # Attributes
        ANNOTATIONS_SUBDIR: Relative path with respect to the root path where
            the ground truth masks are stored. (Annotations)
        SCRIBBLES_SUBDIR: Relative path with respect to the root path where the
            scribbles are stored. (Scribbles)
        RESOLUTION: Resolution of the dataset used to perform all the
            evaluation. (480p)
        sets: Dictionary. The keys are all the DAVIS dataset subsets and the
            values are the list of sequences belonging to that subset.
        dataset: Dictionary. Contains all the information for the entire
            dataset.
            The key is the sequence name and the value is a dictionary of
            informations such as number of frames, number of objects, etc.
        years: List. List with all the versions available from the dataset.

    # Raises
        ValueError: if neither `davis_root` or environmental variable
            `DATASET_DAVIS` are specified.
    """

    # Download information
    # pylint: disable=line-too-long
    SCRIBBLES_URL = 'https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip'
    SCRIBBLES_HASH = '6c6811c67ef757091212a98b68b841305f92b57f6cd2938e0fa94ae8591c3226'
    # pylint: enable=line-too-long

    VERSION = '2017'
    IMAGES_SUBDIR = 'JPEGImages'
    ANNOTATIONS_SUBDIR = 'Annotations'
    SCRIBBLES_SUBDIR = 'Scribbles'
    RESOLUTION = '480p'

    sets = _SETS
    dataset = _DATASET['sequences']
    years = _DATASET['years']

    def __init__(self, davis_root=None):
        self.davis_root = davis_root or os.environ.get('DATASET_DAVIS')
        if self.davis_root is None:
            raise ValueError(
                'Davis root dir not especified. Please specify it in the '
                'environmental variable DAVIS_DATASET or give it as parameter '
                'in davis_root.')
        self.davis_root = Path(self.davis_root).expanduser()
        if self.davis_root.name != 'DAVIS':
            raise ValueError('Davis root folder must be named "DAVIS"')

        if not self.davis_root.exists():
            logging.warning('DAVIS root path do not exists. Creating path.')
            self.davis_root.mkdir(parents=True)

    def _download_scribbles(self):
        file_name = Path(self.SCRIBBLES_URL).name
        download_file = Path(tempfile.mkdtemp()) / file_name

        # Downloading
        logging.info('Downloading Scribbles')
        urllib.request.urlretrieve(self.SCRIBBLES_URL, str(download_file))

        # Check integrity
        logging.info('Checking hash')
        md5 = hashlib.sha256(download_file.read_bytes()).hexdigest()
        if md5 != self.SCRIBBLES_HASH:
            raise ValueError(('Downloaded file do not have a correct hash.\n'
                              'Expected {}\n'
                              'Obtained: {}').format(self.SCRIBBLES_HASH, md5))

        # Extract file
        logging.info('Extracting file')
        zipfile.ZipFile(download_file.open(mode='rb')).extractall(
            str(self.davis_root.parent))
        download_file.unlink()
        logging.info('Download completed')

    def check_files(self, sequences):
        """ Check if the required files are found on DAVIS root.

        Check if all the annotations and scribbles files, required to do the
        evaluation are found on `davis_root`.
        If the scribbles files are not found, it downloads them from the
        internet.

        # Arguments
            sequences: List. List of sequences you want to check.

        # Raises
            FileNotFoundError: if any required files is not found.
        """
        for seq in sequences:
            seq_scribbles_path = self.davis_root.joinpath(
                Davis.SCRIBBLES_SUBDIR, seq)
            seq_annotations_path = self.davis_root.joinpath(
                Davis.ANNOTATIONS_SUBDIR, Davis.RESOLUTION, seq)

            # Check scribbles files needed to give them as base for the user
            nb_scribbles = self.dataset[seq]['num_scribbles']
            for i in range(1, nb_scribbles + 1):
                scribble_file = seq_scribbles_path / '{:03d}.json'.format(i)
                if not scribble_file.exists():
                    self._download_scribbles()

            # Check annotations files required for the evaluation
            nb_frames = self.dataset[seq]['num_frames']
            for i in range(nb_frames):
                annotation_file = seq_annotations_path / '{:05}.png'.format(i)
                if not annotation_file.exists():
                    raise FileNotFoundError(('Annotations file not found for '
                                             'sequence {} and frame {}').format(
                                                 seq, i))

        return True

    def load_scribble(self, sequence, scribble_idx):
        """ Load the scribble from given sequence specifying its index.

        # Arguments
            sequence: String. Sequence name.
            scribble_idx: Integer. Index of the scribble to load.

        # Returns
            Dictionary: Scribble data stored in a dictionary with its default
                format.
        """
        scribble_file = self.davis_root.joinpath(
            Davis.SCRIBBLES_SUBDIR, sequence,
            '{:03d}.json'.format(scribble_idx))

        with scribble_file.open() as fp:
            scribble_data = json.load(fp)
        assert scribble_data['sequence'] == sequence

        logging.verbose(
            'Loaded scribble for sequence {} and scribble_idx {}'.format(
                sequence, scribble_idx), 1)
        logging.verbose(scribble_file, 2)

        return scribble_data

    def load_annotations(self, sequence, dtype=np.int):
        """ Load the annotations of the specified sequence.

        # Arguments
            sequence: String. Sequence name.
            dtype: Numpy Data Type. Data type to return the annotations.
                Default value is `np.int`.

        # Returns
            Numpy Array: Array with the annotations of the given sequence. The
                shape of the array will be `(nb_frames x H x W)` and the value
                will be the index of the objects, being `0` the background.
        """
        root_path = self.davis_root.joinpath(Davis.ANNOTATIONS_SUBDIR,
                                             Davis.RESOLUTION, sequence)
        num_frames = self.dataset[sequence]['num_frames']
        img_size = self.dataset[sequence]['image_size']

        annotations = np.empty(
            (num_frames, img_size[1], img_size[0]), dtype=dtype)

        for f in range(num_frames):
            ann_path = root_path / '{:05d}.png'.format(f)
            mask = Image.open(ann_path)
            mask = np.asarray(mask)
            assert mask.shape == tuple(img_size[::-1])
            annotations[f] = mask

        logging.verbose('Loaded annotations for sequence %s' % sequence, 1)
        logging.verbose('at path: %s' % root_path, 2)
        logging.verbose('Annotations shape: {}'.format(annotations.shape), 2)

        return annotations

    def load_images(self, sequence, dtype=np.uint8):
        """ Load the images of the specified sequence.

        # Arguments
            sequence: String. Sequence name.
            dtype: Numpy Data Type. Data type to return the images. Default
                value is `np.uint8`.

        # Returns
            Numpy Array: Array with all images of the given sequence. The shape
                of the array will be `(nb_frames x H x W x 3)` and the value
                will be the pixel's value with range: `[0, 255]`.
        """
        root_path = self.davis_root.joinpath(Davis.IMAGES_SUBDIR,
                                             Davis.RESOLUTION, sequence)
        num_frames = self.dataset[sequence]['num_frames']
        img_size = self.dataset[sequence]['image_size']

        images = np.empty(
            (num_frames, img_size[1], img_size[0], 3), dtype=dtype)

        for f in range(num_frames):
            img_path = root_path / '{:05d}.jpg'.format(f)
            img = Image.open(img_path)
            img = np.asarray(img)
            assert img.shape[:2] == tuple(img_size[::-1])
            assert img.shape[-1] == 3
            images[f] = img

        logging.verbose('Loaded images for sequence %s' % sequence, 1)
        logging.verbose('at path: %s' % root_path, 2)
        logging.verbose('Annotations shape: {}'.format(images.shape), 2)

        return images
