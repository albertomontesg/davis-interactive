import numpy as np
from scipy.special import comb


def bezier_curve(points, nb_points=1000):
    """ Given a list of points compute a bezier curve from it

    Args:
        points (ndarray): Array of points with shape (N, 2) being N the
            number of points and the second dimension representing the
            (x, y) coordinates.
        nb_points (int): Number of points to sample from the bezier curve.
            This value must be larger than the number of points given in
            `points`.

    Returns:
        (ndarray): Array of shape (1000, 2) with the bezier curve of the
            given path of points.

    """
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
