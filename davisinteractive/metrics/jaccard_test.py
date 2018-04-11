from __future__ import absolute_import, division

import numpy as np
import pytest

from .jaccard import batched_jaccard


class TestJaccard:

    @pytest.mark.parametrize('nb_objects,nb_frames,height,width',
                             [(1, 10, 400, 800), (2, 10, 400, 800),
                              (10, 10, 400, 800), (3, 2, 100, 100),
                              (3, 100, 100, 100)])
    def test_jaccard(self, nb_objects, nb_frames, height, width):
        y_true = np.random.randint(
            0, nb_objects + 1, size=(nb_frames, height, width), dtype=np.int)
        y_pred = np.random.randint(
            0, nb_objects + 1, size=(nb_frames, height, width), dtype=np.int)

        jaccard_frames = batched_jaccard(
            y_true, y_pred, average_over_objects=True)

        assert jaccard_frames.dtype == np.float
        assert jaccard_frames.shape == (nb_frames,)
        assert jaccard_frames.min() >= 0.
        assert jaccard_frames.max() <= 1.

        jaccard_objects = batched_jaccard(
            y_true, y_pred, average_over_objects=False)

        assert jaccard_objects.dtype == np.float
        assert jaccard_objects.shape == (nb_frames, nb_objects)
        assert jaccard_objects.min() >= 0.
        assert jaccard_objects.max() <= 1.

        # Test empty frame
        frame = nb_frames // 2
        y_true[frame] = 0
        y_pred[frame] = 0
        assert np.all(y_true[frame] == 0)
        assert np.all(y_pred[frame] == 0)
        jaccard_frames = batched_jaccard(
            y_true, y_pred, average_over_objects=True)
        assert jaccard_frames[frame] == 1.
        assert not np.any(np.isnan(jaccard_frames))
        jaccard_objects = batched_jaccard(
            y_true, y_pred, average_over_objects=False)
        assert np.all(jaccard_objects[frame, :] == 1.)
        assert not np.any(np.isnan(jaccard_objects))
