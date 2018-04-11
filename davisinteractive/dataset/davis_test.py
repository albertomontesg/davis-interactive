from __future__ import absolute_import, division

import os

import pytest

from .davis import Davis

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


class TestDavis:

    def test_path(self):
        with pytest.raises(ValueError):
            davis = Davis()

        with patch.dict('os.environ', {'DATASET_DAVIS': '/tmp'}):
            davis = Davis()
        davis = Davis(davis_root='/tmp')

    def test_data(self):
        davis = Davis(davis_root='/tmp')

        subsets = {'train': 60, 'val': 30, 'test-dev': 30}
        total = 0
        for s, l in subsets.items():
            assert s in davis.sets
            assert len(davis.sets[s]) == l
            assert s in davis.dataset['sets']
            total += l

        for y in [2016, 2017]:
            assert y in davis.dataset['years']

        assert len(davis.dataset['sequences']) == total
