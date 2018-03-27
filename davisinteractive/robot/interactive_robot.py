import time

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.special import comb
from skimage.morphology import medial_axis
from sklearn.neighbors import radius_neighbors_graph

from ..metrics import batched_jaccard
from ..utils.operations import bezier_curve

__all__ = ['InteractiveScribblesRobot']


class InteractiveScribblesRobot(object):
    def __init__(self, kernel_size=.2, min_nb_nodes=4, nb_points=1000):
        # Kernel size proportional to squared area
        self.kernel_size = kernel_size
        # To prune very small scribbles
        self.min_nb_nodes = min_nb_nodes
        # Number of points to interpolate the bezier curves
        self.nb_points = nb_points

    def _generate_scribble_mask(self, mask):
        """ Generate the skeleton from a mask
        Given an error mask, the medial axis is computed to obtain the
        skeleton of the objects. In order to obtain smoother skeleton and
        remove small objects, an erosion and dilation operations are performed.
        The kernel size used is proportional the squared of the area.

        Args:
            mask (ndarray): Error mask

        Returns:
            skel (ndarray): Skeleton mask
        """
        mask = np.asarray(mask)
        side = np.sqrt(np.sum(mask > 0))

        # Remove small objects and small holes
        mask_ = mask.copy().astype(np.uint8)
        kernel_size = int(self.kernel_size * side)
        compute = True
        while kernel_size > 0 and compute:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
            mask_ = cv2.erode(
                mask.copy().astype(np.uint8), kernel, iterations=1)
            mask_ = cv2.dilate(mask_, kernel, iterations=1)
            compute = False
            if mask_.astype(np.bool).sum() == 0:
                compute = True
                kernel_size = int(kernel_size * .95)
                print('Reducing kernel size')

        skel = medial_axis(mask_.astype(np.bool))
        return skel

    def _mask2graph(self, skeleton_mask):
        """ Transforms a skeleton mask into a graph

        Args:
            skeleton_mask (ndarray): Skeleton mask

        Returns:
            tuple(nx.Graph, ndarray): Returns a tuple where the first element
                is a Graph and the second element is an array of xy coordinates
                indicating the coordinates for each Graph node.

                If an empty mask is given, None is returned.
        """
        mask = np.asarray(skeleton_mask, dtype=np.bool)
        if np.sum(mask) == 0:
            return None

        h, w = mask.shape
        x, y = np.arange(w), np.arange(h)
        X, Y = np.meshgrid(x, y)

        X, Y = X.ravel(), Y.ravel()
        M = mask.ravel()

        X, Y = X[M], Y[M]
        points = np.c_[X, Y]
        G = radius_neighbors_graph(points, np.sqrt(2), mode='distance')
        T = nx.from_scipy_sparse_matrix(G)

        return T, points

    def _acyclics_subgraphs(self, G):
        """ Divide a graph into connected components subgraphs
        Divide a graph into connected components subgraphs and remove its
        cycles removing the edge with higher weight inside the cycle. Also
        prune the graphs by number of nodes in case the graph has not enought
        nodes.

        Args:
            G (nx.Graph): Graph

        Returns:
            list(nx.Graph): Returns a list of graphs which are subgraphs of G
                with cycles removed.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        S = []  # List of subgraphs of G

        for g in nx.connected_component_subgraphs(G):

            # Remove all cycles that we may find
            has_cycles = True
            while has_cycles:
                try:
                    cycle = nx.find_cycle(g)
                    weights = np.asarray([G[u][v]['weight'] for u, v in cycle])
                    idx = weights.argmax()
                    # Remove the edge with highest weight at cycle
                    g.remove_edge(*cycle[idx])
                except nx.NetworkXNoCycle:
                    has_cycles = False

            if len(g) < self.min_nb_nodes:
                # Prune small subgraphs
                continue

            S.append(g)

        return S

    def _longest_path_in_tree(self, G):
        """ Given a tree graph, compute the longest path and return it
        Given an undirected tree graph, compute the longest path and return it.

        The approach use two shortest path transversals (shortest path in a
        tree is the same as longest path). This could be improve but would
        require implement it:
        https://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal

        Args:
            G (nx.Graph): Graph which should be an undirected tree graph

        Returns:
            list(int): Returns a list of indexes of the nodes belonging to the
                longest path.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        if not nx.is_tree(G):
            raise ValueError('Graph G must be a tree (graph without cycles)')

        # Compute the furthest node to the random node v
        v = list(G.nodes())[0]
        distance = nx.single_source_shortest_path_length(G, v)
        vp = max(distance.items(), key=lambda x: x[1])[0]
        # From this furthest point v' find again the longest path from it
        distance = nx.single_source_shortest_path(G, vp)
        longest_path = max(distance.values(), key=len)
        # Return the longest path

        return list(longest_path)

    def interact(self, sequence, pred_masks, gt_masks):
        """ Interaction of the Scribbles robot given a prediction.
        Given the sequence and a mask prediction, the robot will return a
        scribble in the worst path.

        Args:
            sequence (string): Name of the sequence to interact with
            pred_masks (ndarray): Array with the prediction masks. It must be an
                integer array with shape (B x H x W) being B the number of
                frames of the sequence.
            gt_masks (ndarray): Array with the ground truth of the sequence. It
                must have the same data type and shape as `pred_masks`

        Returns:
            (dict): Return a scribble on its default representation.
        """

        predictions = np.asarray(pred_masks, dtype=np.int)
        annotations = np.asarray(gt_masks, dtype=np.int)

        # Infer height and width of the sequence
        h, w = annotations.shape[1:3]
        img_shape = np.asarray([w, h], dtype=np.float)

        jac = batched_jaccard(annotations, predictions)
        worst_frame = jac.argmin()
        pred, gt = predictions[worst_frame], annotations[worst_frame]

        nb_frames = len(annotations)
        obj_ids = np.unique(annotations[annotations < 255])

        scribbles = [[] for _ in range(nb_frames)]

        for obj_id in obj_ids:
            start_time = time.time()
            error_mask = (gt == obj_id) & (pred != obj_id)
            if error_mask.sum() == 0:
                continue

            # Generate scribbles
            skel_mask = self._generate_scribble_mask(error_mask)
            if skel_mask.sum() == 0:
                continue
            G, P = self._mask2graph(skel_mask)
            S = self._acyclics_subgraphs(G)
            longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
            longest_paths = [P[idx] for idx in longest_paths_idx]
            scribbles_paths = [
                bezier_curve(p, self.nb_points) for p in longest_paths
            ]
            end_time = time.time()
            # Generate scribbles data file
            for p in scribbles_paths:
                p /= img_shape
                path_data = {
                    'path': p.tolist(),
                    'object_id': obj_id,
                    'start_time': start_time,
                    'end_time': end_time
                }
                scribbles[worst_frame].append(path_data)

        scribbles_data = {
            'scribbles': scribbles,
            'sequence': sequence,
            'annotated_frame': worst_frame
        }
        return scribbles_data
