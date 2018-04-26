from __future__ import absolute_import, division

import numpy as np
from scipy.special import comb


def bezier_curve(points, nb_points=1000):
    """ Given a list of points compute a bezier curve from it.

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the
            number of points and the second dimension representing the
            (x, y) coordinates.
        nb_points: Integer. Number of points to sample from the bezier curve.
            This value must be larger than the number of points given in
            `points`. Maximum value 10000.

    # Returns
        ndarray: Array of shape (1000, 2) with the bezier curve of the
            given path of points.

    """
    nb_points = min(nb_points, 1000)

    points = np.asarray(points, dtype=np.float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            '`points` should be two dimensional and have shape: (N, 2)')

    n_points = len(points)
    if n_points > nb_points:
        # We are downsampling points
        return points

    t = np.linspace(0., 1., nb_points).reshape(1, -1)

    # Compute the Bernstein polynomial of n, i as a function of t
    i = np.arange(n_points).reshape(-1, 1)
    n = n_points - 1
    polynomial_array = comb(n, i) * (t**(n - i)) * (1 - t)**i

    bezier_curve_points = polynomial_array.T.dot(points)

    return bezier_curve_points


def bresenham(points):
    """ Apply Bresenham algorithm for a list points.

    More info: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the number
            if points and the second coordinate representing the (x, y)
            coordinates.

    # Returns
        ndarray: Array of points after having applied the bresenham algorithm.
    """

    points = np.asarray(points, dtype=np.int)

    def line(x0, y0, x1, y1):
        """ Bresenham line algorithm.
        """
        d_x = x1 - x0
        d_y = y1 - y0

        x_sign = 1 if d_x > 0 else -1
        y_sign = 1 if d_y > 0 else -1

        d_x = np.abs(d_x)
        d_y = np.abs(d_y)

        if d_x > d_y:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            d_x, d_y = d_y, d_x
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * d_y - d_x
        y = 0

        line = np.empty((d_x + 1, 2), dtype=points.dtype)
        for x in range(d_x + 1):
            line[x] = [x0 + x * xx + y * yx, y0 + x * xy + y * yy]
            if D >= 0:
                y += 1
                D -= 2 * d_x
            D += 2 * d_y

        return line

    nb_points = len(points)
    if nb_points < 2:
        return points

    new_points = []

    for i in range(nb_points - 1):
        p = points[i:i + 2].ravel().tolist()
        new_points.append(line(*p))

    new_points = np.concatenate(new_points, axis=0)

    return new_points
