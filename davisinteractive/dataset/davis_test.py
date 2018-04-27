from __future__ import absolute_import, division

import os
import tempfile
import unittest

import numpy as np
import pytest

from davisinteractive.common import Path, patch
from davisinteractive.dataset import Davis
from davisinteractive.utils.scribbles import annotated_frames, is_empty

# Python2/3 Compatibility
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError  # pylint: disable=redefined-builtin

FIXTURE_DIR = os.path.join(tempfile.mkdtemp(), 'DAVIS')


class TestDavis(unittest.TestCase):

    def test_properties(self):
        subsets = {'train': 60, 'val': 30, 'test-dev': 30, 'trainval': 90}

        sequences = []
        for subset, count in subsets.items():
            assert subset in Davis.sets
            assert len(Davis.sets[subset]) == count
            if subset != 'trainval':
                sequences += Davis.sets[subset]

        assert len(sequences) == 120
        assert len(Davis.dataset) == 120

        for s in sequences:
            assert s in Davis.dataset

    def test_path(self):
        with pytest.raises(ValueError):
            davis = Davis()

        with pytest.raises(ValueError):
            davis = Davis(davis_root='/tmp')

        with patch.dict('os.environ', {'DATASET_DAVIS': '/tmp/DAVIS'}):
            davis = Davis()
        davis = Davis(davis_root='/tmp/DAVIS')

    def test_checking_fail(self):
        davis = Davis(FIXTURE_DIR)
        with pytest.raises(FileNotFoundError):
            davis.check_files(Davis.sets['train'])

    def test_load_scribble(self):
        dataset_dir = Path(__file__).parent / 'test_data' / 'DAVIS'

        davis = Davis(dataset_dir)
        scribble = davis.load_scribble('bear', 1)

        assert scribble['sequence'] == 'bear'
        assert not is_empty(scribble)
        assert annotated_frames(scribble) == [39]

    def test_load_annotations(self):
        dataset_dir = Path(__file__).parent / 'test_data' / 'DAVIS'

        num_frames = Davis.dataset['bear']['num_frames']
        Davis.dataset['bear']['num_frames'] = 1

        davis = Davis(dataset_dir)
        ann = davis.load_annotations('bear')

        assert ann.shape == (1, 480, 854)
        assert ann.dtype == np.int
        assert np.all(np.unique(ann) == np.asarray([0, 1]))

        ann2 = davis.load_annotations('bear', dtype=np.uint8)
        assert ann2.shape == (1, 480, 854)
        assert ann2.dtype == np.uint8
        assert np.all(np.unique(ann2) == np.asarray([0, 1]))
        assert np.all(ann2.astype(np.int) == ann)

        Davis.dataset['bear']['num_frames'] = num_frames

    def test_load_images(self):
        dataset_dir = Path(__file__).parent / 'test_data' / 'DAVIS'

        num_frames = Davis.dataset['bear']['num_frames']
        Davis.dataset['bear']['num_frames'] = 1

        davis = Davis(dataset_dir)
        img = davis.load_images('bear')

        assert img.shape == (1, 480, 854, 3)
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255

        img2 = davis.load_images('bear', np.int)
        assert img2.shape == (1, 480, 854, 3)
        assert img2.dtype == np.int
        assert img2.min() >= 0
        assert img2.max() <= 255
        assert np.all(img.astype(np.int) == img2)

        Davis.dataset['bear']['num_frames'] = num_frames
