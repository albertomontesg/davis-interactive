from __future__ import absolute_import, division

import os
import tempfile
import unittest

import pandas as pd
import pytest

from davisinteractive.common import patch
from davisinteractive.connector.local import LocalConnector
from davisinteractive.dataset import Davis
from davisinteractive.session import DavisInteractiveSession
from davisinteractive.utils.scribbles import is_empty

EMPTY_SCRIBBLE = {
    'scribbles': [[] for _ in range(69)],
    'sequence': 'test-sequence'
}


class TestDavisInteractiveSession(unittest.TestCase):

    @patch.object(Davis, 'check_files', return_value=None)
    def test_subset(self, mock_davis):
        davis_root = '/tmp/DAVIS'

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

    @patch.object(
        LocalConnector, 'post_predicted_masks', return_value=EMPTY_SCRIBBLE)
    @patch.object(LocalConnector, 'get_report', return_value=pd.DataFrame())
    @patch.object(LocalConnector, 'get_scribble', return_value=EMPTY_SCRIBBLE)
    @patch.object(
        LocalConnector,
        'get_samples',
        return_value=([('bear', 2), ('bear', 1)], 5, None))
    def test_interactions_limit(self, mock_start_session, mock_get_scribble,
                                mock_get_report, mock_submit_masks):
        davis_root = '/tmp/DAVIS'

        with DavisInteractiveSession(
                davis_root=davis_root,
                max_nb_interactions=5,
                report_save_dir=tempfile.mkdtemp(),
                max_time=None) as session:

            assert mock_start_session.call_count == 1

            for i in range(7):
                assert session.next()
                seq, scribbles, new_seq = session.get_scribbles()
                assert seq == 'bear'
                assert is_empty(scribbles)
                if i % 5 == 0:
                    assert new_seq, i

                session.submit_masks(None)

        assert mock_get_scribble.call_count == 2
        assert mock_submit_masks.call_count == 7
