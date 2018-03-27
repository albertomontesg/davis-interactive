import numpy as np
import pytest

from .scribbles import fuse_scribbles, scribbles2mask, scribbles2points


class TestScribbles2Mask:
    def test_resolution(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
            'annotated_frame':
            1
        }

        mask = scribbles2mask(scribbles_data, (480, 856))
        assert mask.shape == (2, 480, 856)
        assert mask.dtype == np.int

        mask = scribbles2mask(scribbles_data, (100, 100))
        assert mask.shape == (2, 100, 100)
        assert mask.dtype == np.int

        mask = scribbles2mask(scribbles_data, (1, 1))
        assert mask.shape == (2, 1, 1)
        assert mask.dtype == np.int

        with pytest.raises(ValueError):
            mask = scribbles2mask(scribbles_data, (0, 100))
        with pytest.raises(ValueError):
            mask = scribbles2mask(scribbles_data, (1, -5))
        with pytest.raises(ValueError):
            mask = scribbles2mask(scribbles_data, (1, 100, 100))

    def test_output(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
            'annotated_frame':
            1
        }
        mask = scribbles2mask(scribbles_data, (480, 856))
        assert mask.shape == (2, 480, 856)
        assert mask.dtype == np.int

        mask = scribbles2mask(
            scribbles_data, (480, 856), only_annotated_frame=True)
        assert mask.shape == (480, 856)
        assert mask.dtype == np.int
        assert np.any(mask != -1)

    def test_mask(self):
        pass


class TestScribbles2Points:
    def test_wo_output_resolution(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
            'annotated_frame':
            1
        }

        X, Y = scribbles2points(scribbles_data)
        assert X.shape == (2, 2)
        assert X.dtype == np.float
        assert Y.shape == (2, )
        assert Y.dtype == np.int

        assert np.all(X == np.asarray([[0, 0], [0, 0.1]]))
        assert np.all(Y == np.asarray([1, 1]))

    def test_w_output_resolution(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
            'annotated_frame':
            1
        }

        X, Y = scribbles2points(scribbles_data, output_resolution=(100, 100))
        assert X.shape == (2, 2)
        assert X.dtype == np.int
        assert Y.shape == (2, )
        assert Y.dtype == np.int

        assert np.all(X == np.asarray([[0, 0], [0, 9]], dtype=np.int))
        assert np.all(Y == np.asarray([1, 1]))

        X, Y = scribbles2points(scribbles_data, output_resolution=(101, 101))
        assert np.all(X == np.asarray([[0, 0], [0, 10]], dtype=np.int))
