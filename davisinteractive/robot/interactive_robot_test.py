from __future__ import absolute_import, division

import networkx as nx
import numpy as np

from . import InteractiveScribblesRobot


class TestInteractiveScribblesRobot:

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
