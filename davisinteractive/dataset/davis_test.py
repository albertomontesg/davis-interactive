from __future__ import absolute_import, division

import os
import tempfile
import unittest

import pytest

from davisinteractive.dataset import Davis

# Python2/3 Compatibility
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError  # pylint: disable=redefined-builtin

FIXTURE_DIR = os.path.join(tempfile.mkdtemp(), 'DAVIS')


class TestDavis(unittest.TestCase):

    def test_properties(self):
        subsets = {'train': 60, 'val': 30, 'test-dev': 30}

        sequences = []
        for subset, count in subsets.items():
            assert subset in Davis.sets
            assert len(Davis.sets[subset]) == count
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
