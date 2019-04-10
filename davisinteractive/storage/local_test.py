import unittest

import numpy as np
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
        contour = [.8, .6, .4]

        storage = LocalStorage()

        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, [.1, .2, 1.0001], contour)
        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, [-.1, .2, 1], contour)
        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, [1, 1], jaccard, contour)
        with pytest.raises(ValueError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, jaccard, [-0.01, 1.0, .4])

        assert storage.store_interactions_results(
            user_id, session_id, sequence, scribble_idx, interaction, timing,
            objects_idx, frames, [.1, .000, 1.000], contour)
        with pytest.raises(RuntimeError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction,
                timing, objects_idx, frames, jaccard, contour)
        with pytest.raises(RuntimeError):
            storage.store_interactions_results(
                user_id, session_id, sequence, scribble_idx, interaction + 2,
                timing, objects_idx, frames, jaccard, contour)
        assert storage.store_interactions_results(
            user_id, session_id, sequence, scribble_idx, interaction + 1,
            timing, objects_idx, frames, jaccard, contour)

    def test_annotated_frames(self):
        session_id = 'unused'
        sequence = 'bmx-trees'
        scribble_idx = 1

        storage = LocalStorage()
        storage.store_annotated_frame(session_id, sequence, scribble_idx, 1,
                                      False)
        annotated_frames = storage.get_annotated_frames(session_id, sequence,
                                                        scribble_idx)
        self.assertEqual(annotated_frames, (1,))

    def test_annotated_frames_full(self):
        session_id = 'unused'
        sequence = 'bmx-trees'
        scribble_idx = 1
        nb_frames = 80

        storage = LocalStorage()
        for i in range(nb_frames):
            storage.store_annotated_frame(session_id, sequence, scribble_idx, i,
                                          False)
        annotated_frames = storage.get_annotated_frames(session_id, sequence,
                                                        scribble_idx)
        self.assertEqual(annotated_frames, tuple())
