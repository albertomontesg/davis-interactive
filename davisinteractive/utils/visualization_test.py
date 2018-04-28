from __future__ import absolute_import, division

import unittest

import numpy as np
import pytest

from davisinteractive.utils.visualization import (_pascal_color_map,
                                                  overlay_mask, plot_scribble)

import matplotlib  # isort:skip
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # isort:skip


class TestPascalColormap(unittest.TestCase):

    def test_default(self):
        colormap = _pascal_color_map()
        assert colormap.shape == (256, 3)
        assert colormap.dtype == np.uint8

        assert np.all(colormap[0] == [0, 0, 0])
        assert np.all(colormap[1] == [128, 0, 0])
        assert np.all(colormap[2] == [0, 128, 0])
        assert np.all(colormap[3] == [128, 128, 0])
        assert np.all(colormap[255] == [224, 224, 192])

    def test_normalized(self):
        colormap = _pascal_color_map(normalized=True)
        assert colormap.shape == (256, 3)
        assert colormap.dtype == np.float32

        assert np.all(np.isclose(colormap[0], [0, 0, 0]))
        assert np.all(np.isclose(colormap[1], [128 / 255, 0, 0]))
        assert np.all(np.isclose(colormap[2], [0, 128 / 255, 0]))
        assert np.all(np.isclose(colormap[3], [128 / 255, 128 / 255, 0]))
        assert np.all(
            np.isclose(colormap[255], [224 / 255, 224 / 255, 192 / 255]))


class TestPlotScribble(unittest.TestCase):

    def test_plot(self):
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
        _, ax = plt.subplots(1)
        plot_scribble(ax, scribble, 1, output_size=(100, 200))

        plot_scribble(ax, scribble, 1)

        with pytest.raises(ValueError):
            plot_scribble(ax, scribble, 10)


class TestOverlay(unittest.TestCase):

    def test_shape_mismatch(self):
        im = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        ann = np.zeros((100, 201))

        with pytest.raises(ValueError):
            overlay_mask(im, ann)

    def test_wrong_channel_dimensions(self):
        im = np.random.randint(0, 255, (100, 200, 4), dtype=np.uint8)
        ann = np.zeros((100, 200))

        with pytest.raises(ValueError):
            overlay_mask(im, ann)

    def test_empty_mask(self):
        im = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        ann = np.zeros((100, 200))

        img = overlay_mask(im, ann)

        assert img.shape == im.shape
        assert img.dtype == im.dtype
        assert np.all(img == im)

    def test_full_mask(self):
        im = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        ann = np.ones((100, 200))

        img = overlay_mask(im, ann, alpha=1.)

        assert img.shape == im.shape
        assert img.dtype == im.dtype
        assert np.all(img == im)

        img = overlay_mask(im, ann, alpha=0.)

        assert img.shape == im.shape
        assert img.dtype == im.dtype
        assert np.all(img == [128, 0, 0])

        img = overlay_mask(im, ann, alpha=0., colors=[[45, 43, 76], [1, 2, 3]])

        assert img.shape == im.shape
        assert img.dtype == im.dtype
        assert np.all(img == [1, 2, 3])

    def test_mask(self):
        im = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        ann = np.zeros((100, 200))
        ann[50:70, 50:150] = 1

        img = overlay_mask(im, ann, alpha=.5)

        assert img.shape == im.shape
        assert img.dtype == im.dtype
        assert np.all(img[ann == 0] == im[ann == 0])
        assert not np.all(img[ann != 0] == im[ann != 0])
