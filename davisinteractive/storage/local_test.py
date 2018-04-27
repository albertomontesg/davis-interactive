import unittest

import pytest

from davisinteractive.storage import LocalStorage


class TestLocalStorage(unittest.TestCase):

    def test_init(self):
        storage = LocalStorage()

        for c in storage.COLUMNS:
            assert c in storage.report

    def test_store_operation(self):
        user_id = 'empty'
        session_id = '12345'
        sequence = 'test'
        scribble_idx = 1
        interaction = 1
        timing = 10.34
        objects_idx = [1, 2, 3]
        frames = [0, 0, 0]
        jaccard = [.2, .3, .4]

        storage = LocalStorage()

        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, [.1, .2, 1.0001])
        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, [-.1, .2, 1])
        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, [1, 1], jaccard)

        assert storage.store_interactions_results(
            user_id, session_id, sequence, scribble_idx, interaction, timing,
            objects_idx, frames, [.1, .000, 1.000])
        with pytest.raises(RuntimeError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, jaccard)
        with pytest.raises(RuntimeError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction + 2,
                timing, objects_idx, frames, jaccard)
        assert storage.store_interactions_results(
            user_id, session_id, sequence, scribble_idx, interaction + 1,
            timing, objects_idx, frames, jaccard)
