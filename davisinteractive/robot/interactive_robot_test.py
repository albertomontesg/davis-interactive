from __future__ import absolute_import, division

import json
import unittest

import networkx as nx
import numpy as np
import pytest

from davisinteractive.robot import InteractiveScribblesRobot
from davisinteractive.utils.scribbles import annotated_frames, is_empty


class TestInteractiveScribblesRobot(unittest.TestCase):

    def test_generate_scribble_mask_empty(self):
        empty_mask = np.zeros((100, 200), dtype=np.bool)

        robot = InteractiveScribblesRobot()
        skel = robot._generate_scribble_mask(empty_mask)
        assert skel.shape == empty_mask.shape
        assert np.all(skel == empty_mask)

    def test_generate_scribble_mask(self):
        empty_mask = np.zeros((100, 200), dtype=np.bool)
        squared_mask = empty_mask.copy()
        squared_mask[50:100, 100:150] = True

        robot = InteractiveScribblesRobot()
        skel_squared = robot._generate_scribble_mask(squared_mask)
        assert skel_squared.shape == empty_mask.shape
        assert np.sum(skel_squared) > 0

    def test_mask2graph_empty(self):
        empty_mask = np.zeros((100, 200), dtype=np.bool)

        robot = InteractiveScribblesRobot()

        out = robot._mask2graph(empty_mask)
        assert out is None

    def test_mask2graph(self):
        empty_mask = np.zeros((100, 200), dtype=np.bool)
        squared_mask = empty_mask.copy()
        squared_mask[50:100, 100:150] = True

        robot = InteractiveScribblesRobot()
        out = robot._mask2graph(squared_mask)
        assert isinstance(out, tuple)
        assert len(out) == 2
        G, T = out
        assert isinstance(G, nx.Graph)
        assert isinstance(T, np.ndarray)
        assert T.dtype == np.int
        assert len(G) == len(T)
        T_x, T_y = T.T
        assert T_x.min() >= 0
        assert T_x.max() < 200
        assert T_y.min() >= 0
        assert T_y.max() < 100

    def test_interaction_no_class(self):
        gt_empty = np.zeros((10, 300, 500), dtype=np.int)

        robot = InteractiveScribblesRobot()
        with pytest.raises(ValueError):
            robot.interact('test', gt_empty.copy(), gt_empty)

    def test_interaction_equal(self):
        nb_frames, h, w = 10, 300, 500
        gt_empty = np.zeros((nb_frames, h, w), dtype=np.int)
        gt_empty[0, 100:200, 100:200] = 1
        pred_empty = gt_empty.copy()

        robot = InteractiveScribblesRobot()

        scribble = robot.interact('test', pred_empty, gt_empty)
        assert is_empty(scribble)
        assert annotated_frames(scribble) == []
        assert len(scribble['scribbles']) == nb_frames

    def test_interaction(self):
        nb_frames, h, w = 10, 300, 500
        gt_empty = np.zeros((nb_frames, h, w), dtype=np.int)
        pred_empty = gt_empty.copy()
        gt_empty[5, 100:200, 100:200] = 1

        robot = InteractiveScribblesRobot()

        scribble = robot.interact('test', pred_empty, gt_empty)
        assert not is_empty(scribble)
        assert annotated_frames(scribble) == [5]
        assert len(scribble['scribbles']) == nb_frames

        lines = scribble['scribbles'][5]

        for l in lines:
            assert l['object_id'] == 1
            path = np.asarray(l['path'])
            x, y = path[:, 0], path[:, 1]
            assert np.all((x >= .2) & (x <= .4))
            assert np.all((y >= 1 / 3) & (y <= 2 / 3))

    def test_scribble_json_serializer(self):
        nb_frames, h, w = 10, 300, 500
        gt_empty = np.zeros((nb_frames, h, w), dtype=np.int)
        pred_empty = gt_empty.copy()
        gt_empty[5, 100:200, 100:200] = 1

        robot = InteractiveScribblesRobot()

        scribble = robot.interact('test', pred_empty, gt_empty)
        json.JSONEncoder().encode(scribble)
