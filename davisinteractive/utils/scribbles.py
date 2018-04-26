from __future__ import absolute_import, division

import numpy as np

from .operations import bezier_curve
from .operations import bresenham as bresenham_function


def scribbles2mask(scribbles,
                   output_resolution,
                   bezier_curve_sampling=False,
                   nb_points=1000,
                   bresenham=True,
                   default_value=-1):
    """ Convert the scribbles data into a mask.

    # Arguments
        scribbles: Dictionary. Scribbles in the default format.
        output_resolution: Tuple. Output resolution (H, W).
        bezier_curve_sampling: Boolean. Weather to sample first the returned
            scribbles using bezier curve or not.
        nb_points: Integer. If `bezier_curve_sampling` is `True` set the number
            of points to sample from the bezier curve.
        bresenham: Boolean. Whether to compute bresenham algorithm for the
            scribbles lines.
        default_value: Integer. Default value for the pixels which do not belong
            to any scribble.

    # Returns
        ndarray: Array with the mask of the scribbles with the index of the
            object ids. The shape of the returned array is (B x H x W) by
            default or (H x W) if `only_annotated_frame==True`.
    """
    if len(output_resolution) != 2:
        raise ValueError(
            'Invalid output resolution: {}'.format(output_resolution))
    for r in output_resolution:
        if r < 1:
            raise ValueError(
                'Invalid output resolution: {}'.format(output_resolution))

    nb_frames = len(scribbles['scribbles'])
    masks = np.full(
        (nb_frames,) + output_resolution, default_value, dtype=np.int)

    size_array = np.asarray(output_resolution[::-1], dtype=np.float) - 1

    for f in range(nb_frames):
        sp = scribbles['scribbles'][f]
        for p in sp:
            path = p['path']
            obj_id = p['object_id']
            path = np.asarray(path, dtype=np.float)
            if bezier_curve_sampling:
                path = bezier_curve(path, nb_points=nb_points)
            path *= size_array
            path = path.astype(np.int)

            if bresenham:
                path = bresenham_function(path)
            m = masks[f]

            m[path[:, 1], path[:, 0]] = obj_id
            masks[f] = m

    return masks


def scribbles2points(scribbles_data, output_resolution=None):
    """ Convert the given scribbles into a list of points and object ids.

    # Arguments
        scribbles_data: Dictionary. Scribbles in the default format
        output_resolution: Tuple. Output resolution (H, W) to scale the
            points.
            If None given, the points will be floats as a fraction of height
            and width.

    # Returns
        (ndarray, ndarray): Returns (X, Y) where X is a list of points from the
            scribbles represented in the output_resolution with shape (N x 3)
            with N being the total number of points on all the scribbles. The three
            coordinates given correspond the the frame number, height and width,
            respectively.
            Y is the object id for each given point with shape (N,).
    """
    scribbles = scribbles_data['scribbles']

    paths, object_ids = [], []

    for frame, s in enumerate(scribbles):
        for l in s:
            # p = l['path']
            coordinates = [[frame] + point for point in l['path']]
            paths += coordinates
            object_ids += [l['object_id']] * len(l['path'])

    paths = np.asarray(paths, dtype=np.float)
    object_ids = np.asarray(object_ids, dtype=np.int)

    if output_resolution:
        h, w = output_resolution
        img_size = np.asarray([1, h - 1, w - 1], dtype=np.float)
        paths *= img_size
        paths = paths.astype(np.int)

    return paths, object_ids


def fuse_scribbles(scribbles_a, scribbles_b):
    """ Fuse two scribbles in the default format.

    # Arguments
        scribbles_a: Dictionary. Default representation of scribbles A.
        scribbles_b: Dictionary. Default representation of scribbles B.

    # Returns
        dict: Returns a dictionary with scribbles A and B fused.
    """

    if scribbles_a['sequence'] != scribbles_b['sequence']:
        raise ValueError('Scribbles to fuse are not from the same sequence')
    if len(scribbles_a['scribbles']) != len(scribbles_b['scribbles']):
        raise ValueError('Scribbles does not have the same number of frames')

    scribbles = dict(scribbles_a)
    nb_frames = len(scribbles['scribbles'])

    for i in range(nb_frames):
        scribbles['scribbles'][i] += scribbles_b['scribbles'][i]

    return scribbles


def is_empty(scribbles_data):
    """ Checks whether the given scribble has any non-empty line.

    # Arguments
        scribbles_data (dict): Scribble in the default format

    # Returns
        bool: Whether the scribble is empty or not.
    """
    scribbles = scribbles_data['scribbles']
    has_lines = [len(s) > 0 for s in scribbles]
    return not any(has_lines)


def annotated_frames(scribbles_data):
    """ Finds which frames have a scribble.

    # Arguments
        scribbles_data (dict): Scribble in the default format.
    # Returns
        list: Number of the frames that contain at least one scribble.
    """
    scribbles = scribbles_data['scribbles']
    frames_list = [i for i, s in enumerate(scribbles) if s]
    return frames_list


def annotated_frames_object(scribbles_data, object_id):
    """ Computes which frames have a scribble for a certain object.

    # Arguments
        scribbles_data (dict): Scribble in the default format.
        object_id (int): Id of the object of interest.
    # Returns
        dict: Number of the frames that contain at least one scribble.
    """
    frames_list = []
    scribbles = scribbles_data['scribbles']
    for ii, scribble_frame in enumerate(scribbles):
        for scribble in scribble_frame:
            if scribble['object_id'] == object_id:
                frames_list.append(ii)
                break
    return frames_list
