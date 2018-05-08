from __future__ import absolute_import, division

import numpy as np

__all__ = ['batched_jaccard']


def batched_jaccard(y_true, y_pred, average_over_objects=True, nb_objects=None):
    """ Batch jaccard similarity for multiple instance segmentation.

    Jaccard similarity over two subsets of binary elements $A$ and $B$:

    $$
    \mathcal{J} = \\frac{A \\cap B}{A \\cup B}
    $$

    # Arguments
        y_true: Numpy Array. Array of shape (B x H x W) and type integer giving the
            ground truth of the object instance segmentation.
        y_pred: Numpy Array. Array of shape (B x H x W) and type integer giving the
            prediction of the object segmentation.
        average_over_objects: Boolean. Weather or not to average the jaccard over
            all the objects in the sequence. Default True.
        nb_objects: Integer. Number of objects in the ground truth mask. If
            `None` the value will be infered from `y_true`. Setting this value
            will speed up the computation.

    # Returns
        ndarray: Returns an array of shape (B) with the average jaccard for
            all instances at each frame if `average_over_objects=True`. If
            `average_over_objects=False` returns an array of shape (B x nObj)
            with nObj being the number of objects on `y_true`.
    """
    y_true = np.asarray(y_true, dtype=np.int)
    y_pred = np.asarray(y_pred, dtype=np.int)
    if y_true.ndim != 3:
        raise ValueError('y_true array must have 3 dimensions.')
    if y_pred.ndim != 3:
        raise ValueError('y_pred array must have 3 dimensions.')
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')

    if nb_objects is None:
        objects_ids = np.unique(y_true[(y_true < 255) & (y_true > 0)])
        nb_objects = len(objects_ids)
    else:
        objects_ids = [i + 1 for i in range(nb_objects)]
        objects_ids = np.asarray(objects_ids, dtype=np.int)
    if nb_objects == 0:
        raise ValueError('Number of objects in y_true should be higher than 0.')
    nb_frames = len(y_true)

    jaccard = np.empty((nb_frames, nb_objects), dtype=np.float)

    for i, obj_id in enumerate(objects_ids):
        mask_true, mask_pred = y_true == obj_id, y_pred == obj_id

        union = (mask_true | mask_pred).sum(axis=(1, 2))
        intersection = (mask_true & mask_pred).sum(axis=(1, 2))

        for j in range(nb_frames):
            if np.isclose(union[j], 0):
                jaccard[j, i] = 1.
            else:
                jaccard[j, i] = intersection[j] / union[j]

    if average_over_objects:
        jaccard = jaccard.mean(axis=1)
    return jaccard
