import numpy as np
import pytest

from .scribbles import fuse_scribbles, scribbles2mask


class TestScribblesUtils:
    def test_scribbles2mask_resolution(self):
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

    def test_scribbles2mask_output(self):
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

    def test_scribbles2mask_mask(self):
        pass
