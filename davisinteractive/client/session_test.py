import os
from unittest.mock import patch

import pytest

from ..dataset import Davis
from .session import DavisInteractiveSession


class TestDavisInteractiveSession:
    def test_subset(self):
        davis_root = '/tmp'

        with patch.object(Davis, 'check_files', return_value=None) as mm:
            session = DavisInteractiveSession(
                subset='val', davis_root=davis_root)
            session.__enter__()
            session = DavisInteractiveSession(
                subset='train', davis_root=davis_root)
            session.__enter__()

            session1 = DavisInteractiveSession(
                subset='test-dev', davis_root=davis_root)
            session2 = DavisInteractiveSession(
                subset='xxxx', davis_root=davis_root)
            with pytest.raises(ValueError):
                session1.__enter__()
            with pytest.raises(ValueError):
                session2.__enter__()

            assert mm.call_count == 2

    # def test_interactions_limit(self):
    #     davis_root = '/tmp'
    #     with patch.object(Davis, 'check_files', return_value=None) as mm:
    #         session = DavisInteractiveSession(
    #             davis_root=davis_root, max_nb_interactions=5)
    #         session.__enter__()
    #         assert mm.call_count == 1

    #     for i in range(5):
    #         seq, scribble, new_seq = session.get_scribbles()
    #         if i == 0:
    #             assert new_seq
