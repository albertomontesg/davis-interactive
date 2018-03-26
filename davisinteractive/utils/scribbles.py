import numpy as np


def scribbles2mask(scribbles,
                   output_resolution,
                   only_annotated_frame=False,
                   default_value=-1):
    """ Convert the scribbles data into a mask.

    Args:
        scribbles (dict): Scribbles in the default format
        output_resolution (tuple): Output resolution (H, W).
        only_annotated_frame (bool): Weather return the mask for all the frames
            or only the mask of the single annotated frame (only valid if one
            frame is annotated). Default False.
        default_value (int): default value for the pixels which do not belong
            to any scribble.

    Returns:
        (ndarray): Array with the mask of the scribbles with the index of the
            object ids. The shape of the returned array is (B x H x W) by
            default or (H x W) if `only_annotated_frame==True`.
    """
    nb_frames = len(scribbles['scribbles'])
    masks = np.full(
        (nb_frames, ) + output_resolution, default_value, dtype=np.int)

    size_array = np.asarray(output_resolution[::-1], dtype=np.float)

    for f in range(nb_frames):
        sp = scribbles['scribbles'][f]
        for p in sp:
            path = p['path']
            obj_id = p['object_id']
            path = np.asarray(path, dtype=np.float)
            path *= size_array
            path = path.astype(np.int)
            m = masks[f]

            m[path[:, 1], path[:, 0]] = obj_id
            masks[f] = m

    if only_annotated_frame:
        if 'annotated_frame' not in scribbles:
            raise ValueError('`annotated_frame` not in scribble')
        annotated_frame = scribbles['annotated_frame']
        return masks[annotated_frame]

    return masks


def fuse_scribbles(scribbles_a, scribbles_b):
    """ Fuse two scribbles in the default format.

    Args:
        scribbles_a (dict): Default representation of scribbles A.
        scribbles_b (dict): Default representation of scribbles B.

    Returns:
        (dict): Return a dictionary being the representation of the addition of
            both scribbles A and B.
    """

    assert scribbles_a['sequence'] == scribbles_b[
        'sequence'], 'Scribbles to fuse not from the same sequence'
    assert len(scribbles_a['scribbles']) == len(scribbles_b[
        'scribbles']), 'Scribbles does not have the same number of frames'

    scribbles = dict(scribbles_a)
    nb_frames = len(scribbles['scribbles'])

    for i in range(nb_frames):
        scribbles['scribbles'][i] += scribbles_b['scribbles'][i]

    return scribbles
