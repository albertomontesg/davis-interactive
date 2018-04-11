from __future__ import absolute_import, division

import os
import unittest

import pytest

from ..connector.local import LocalConnector
from ..dataset import Davis
from ..utils.scribbles import is_empty
from .session import DavisInteractiveSession

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

EMPTY_SCRIBBLE = {
    'scribbles': [[] for _ in range(69)],
    'sequence': 'test-sequence'
}


class TestDavisInteractiveSession(unittest.TestCase):

    @patch.object(Davis, 'check_files', return_value=None)
    def test_subset(self, mock_davis):
        davis_root = '/tmp'

        session = DavisInteractiveSession(subset='val', davis_root=davis_root)
        session.__enter__()
        session = DavisInteractiveSession(subset='train', davis_root=davis_root)
        session.__enter__()

        session1 = DavisInteractiveSession(
            subset='test-dev', davis_root=davis_root)
        session2 = DavisInteractiveSession(subset='xxxx', davis_root=davis_root)
        with pytest.raises(ValueError):
            session1.__enter__()
        with pytest.raises(ValueError):
            session2.__enter__()

        assert mock_davis.call_count == 2

    @patch.object(LocalConnector, 'close', return_value=None)
    @patch.object(LocalConnector, 'submit_masks', return_value=EMPTY_SCRIBBLE)
    @patch.object(
        LocalConnector, 'get_starting_scribble', return_value=EMPTY_SCRIBBLE)
    @patch.object(
        LocalConnector,
        'start_session',
        return_value=([('test-sequence', 2, 2), ('test-sequence', 1, 3)], 5,
                      None))
    def test_interactions_limit(self, mock_start_session,
                                mock_get_starting_scribble, mock_submit_masks,
                                mock_close):
        davis_root = '/tmp'

        with DavisInteractiveSession(
                davis_root=davis_root, max_nb_interactions=5,
                max_time=None) as session:

            assert mock_start_session.call_count == 1

            for i in range(7):
                assert session.next()
                seq, scribbles, new_seq = session.get_scribbles()
                assert seq == 'test-sequence'
                assert is_empty(scribbles)
                if i % 5 == 0:
                    assert new_seq, i

                session.submit_masks(None)

            assert mock_close.call_count == 0

        assert mock_close.call_count == 1
        assert mock_get_starting_scribble.call_count == 2
        assert mock_submit_masks.call_count == 7
