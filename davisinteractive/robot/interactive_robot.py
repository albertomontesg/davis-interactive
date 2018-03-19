import cv2
import networkx as nx
import numpy as np
from scipy.special import comb
from skimage.morphology import medial_axis
from sklearn.neighbors import radius_neighbors_graph

__all__ = ['InteractiveScribblesRobot']


class InteractiveScribblesRobot(object):
    KERNEL_SIZE = .2  # Kernel size proportional to squared area
    MIN_NB_NODES = 4  # To prune very small scribbles
    N_TIMES = 1000  # Number of points to interpolate the bezier curves

    def __init__(self):
        pass

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
        kernel_size = int(self.KERNEL_SIZE * side)
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
            mask_ = cv2.erode(mask_, kernel, iterations=1)
            mask_ = cv2.dilate(mask_, kernel, iterations=1)

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

            if len(g) < self.MIN_NB_NODES:
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

    def _bezier_curve(self, points):
        """ Given a list of points compute a bezier curve from it

        Args:
            points (ndarray): Array of points with shape (N, 2) being N the
                number of points and the second dimension representing the
                (x, y) coordinates.

        Returns:
            (ndarray): Array of shape (1000, 2) with the bezier curve of the
                given path of points.


        """
        points = np.asarray(points, dtype=np.float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                '`points` should be two dimensional and have shape: (N, 2)')

        n_points = len(points)
        t = np.linspace(0., 1., self.N_TIMES).reshape(1, -1)

        # Compute the Bernstein polynomial of n, i as a function of t
        i = np.arange(n_points).reshape(-1, 1)
        n = n_points - 1
        polynomial_array = comb(n, i) * (t**(n - i)) * (1 - t)**i

        bezier_curve = polynomial_array.T.dot(points)

        return bezier_curve
