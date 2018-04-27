from __future__ import absolute_import, division

import unittest

import numpy as np
import pytest

from .scribbles import (annotated_frames, annotated_frames_object,
                        fuse_scribbles, is_empty, scribbles2mask,
                        scribbles2points)


class TestScribbles2Mask(unittest.TestCase):

    def test_resolution(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test'
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
            'test'
        }
        mask = scribbles2mask(scribbles_data, (480, 856))
        assert mask.shape == (2, 480, 856)
        assert mask.dtype == np.int

    def test_mask_value(self):
        scribble_empty = {
            'scribbles': [[], []],
            'sequence': 'test',
        }
        mask_0 = scribbles2mask(scribble_empty, (100, 100), default_value=-1)
        assert np.all(mask_0 == -1)
        mask_1 = scribbles2mask(scribble_empty, (100, 100), default_value=0)
        assert np.all(mask_1 == 0)
        mask_2 = scribbles2mask(scribble_empty, (100, 100), default_value=2)
        assert np.all(mask_2 == 2)
        mask_3 = scribbles2mask(scribble_empty, (100, 100), default_value=1.9)
        assert np.all(mask_3 == 1)

    def test_mask_wo_bresenham(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test'
        }
        mask = scribbles2mask(
            scribbles_data, (100, 150), bresenham=False, default_value=0)
        assert mask.sum() == 2
        assert mask.dtype == np.int
        assert mask.min() == 0 and mask.max() == 1
        assert mask[1, 0, 0] == 1
        assert mask[1, -1, 0] == 1

    def test_mask_w_bresenham(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test'
        }
        mask = scribbles2mask(
            scribbles_data, (100, 150), bresenham=True, default_value=0)
        assert np.all(mask[0] == 0)
        assert np.all(mask[1, :, 0] == 1)
        assert np.all(mask[1, :, 1:] == 0)

    def test_mask_w_brezier(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test'
        }
        mask = scribbles2mask(
            scribbles_data, (100, 150),
            bresenham=False,
            bezier_curve_sampling=True,
            default_value=0)
        assert np.all(mask[0] == 0)
        assert np.all(mask[1, :, 0] == 1)
        assert np.all(mask[1, :, 1:] == 0)


class TestScribbles2Points(unittest.TestCase):

    def test_wo_output_resolution(self):
        scribbles_data = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test'
        }

        X, Y = scribbles2points(scribbles_data)
        assert X.shape == (2, 3)
        assert X.dtype == np.float
        assert Y.shape == (2,)
        assert Y.dtype == np.int

        assert np.all(X == np.asarray([[1, 0, 0], [1, 0, 0.1]]))
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
            'test'
        }

        X, Y = scribbles2points(scribbles_data, output_resolution=(100, 100))
        assert X.shape == (2, 3)
        assert X.dtype == np.int
        assert Y.shape == (2,)
        assert Y.dtype == np.int

        assert np.all(X == np.asarray([[1, 0, 0], [1, 0, 9]], dtype=np.int))
        assert np.all(Y == np.asarray([1, 1]))

        X, Y = scribbles2points(scribbles_data, output_resolution=(101, 101))
        assert np.all(X == np.asarray([[1, 0, 0], [1, 0, 10]], dtype=np.int))


class TestFuseScribbles(unittest.TestCase):

    def test_fuse_different_frame(self):
        scribble_1 = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
        }
        scribble_2 = {
            'scribbles': [[{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }], []],
            'sequence':
            'test',
        }
        scribble_result = {
            'scribbles': [[{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
        }

        assert scribble_result == fuse_scribbles(scribble_1, scribble_2)

    def test_same_frame(self):
        scribble = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
        }
        scribble_result = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }, {
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
        }

        assert scribble_result == fuse_scribbles(scribble, scribble)

    def test_wrong_sequence(self):
        scribble_1 = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'xxxx',
        }
        scribble_2 = {
            'scribbles': [[{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }], []],
            'sequence':
            'test',
        }
        with pytest.raises(ValueError):
            fuse_scribbles(scribble_1, scribble_2)

    def test_wrong_nb_frames(self):
        scribble_1 = {
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
        scribble_2 = {
            'scribbles': [[{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
        }
        with pytest.raises(ValueError):
            fuse_scribbles(scribble_1, scribble_2)


class TestEmpytScribble(unittest.TestCase):

    def test_not_empty(self):
        scribble = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }]],
            'sequence':
            'test',
        }
        assert not is_empty(scribble)

    def test_empty(self):
        scribble = {'scribbles': [[] for _ in range(10)], 'sequence': 'test'}
        assert is_empty(scribble)


class TestAnnotatedFrames(unittest.TestCase):

    def test_annotated_frames(self):
        scribble = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }, {
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }], [], [], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }], []],
            'sequence':
            'test',
        }
        assert annotated_frames(scribble) == [1, 4]


class TestAnnotatedFramesObject(unittest.TestCase):

    def test_annotated_frames(self):
        scribble = {
            'scribbles': [[], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }, {
                'path': [[0, 0], [0, 0.1]],
                'object_id': 1,
                'start_time': 0,
                'end_time': 1000
            }], [], [], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 2,
                'start_time': 0,
                'end_time': 1000
            }, {
                'path': [[0, 0], [0, 0.1]],
                'object_id': 3,
                'start_time': 0,
                'end_time': 1000
            }], [{
                'path': [[0, 0], [0, 0.1]],
                'object_id': 2,
                'start_time': 0,
                'end_time': 1000
            }], []],
            'sequence':
            'test',
        }
        assert annotated_frames_object(scribble, 1) == [1]
        assert annotated_frames_object(scribble, 2) == [4, 5]
        assert annotated_frames_object(scribble, 3) == [4]
