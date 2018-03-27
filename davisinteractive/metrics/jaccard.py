import numpy as np

__all__ = ['batched_jaccard']


def batched_jaccard(y_true, y_pred, average_over_objects=True):
    """ Batch jaccard similarity for multiple instance segmentation.

    Args:
        y_true (ndarray): Array of shape (BxHxW) and type integer giving the
            ground truth of the object instance segmentation.
        y_pred (ndarray): Array of shape (BxHxW) and type integer giving the
            prediction of the object instance segmentation.
        average_over_objects (bool): Weather or not average the jaccard over
            all the objects in the sequence. Default True.
    Returns:
        (ndarray): Returns an array of shape (B) with the average jaccard for
            all instances at each frame if `average_over_objects=True`. If
            `average_over_objects=False` returns an array of shape (B x O)
            being O the number of objects on `y_true`.
    """
    y_true = np.asarray(y_true, dtype=np.int)
    y_pred = np.asarray(y_pred, dtype=np.int)
    if y_true.ndim != 3:
        raise ValueError('y_true array must have 3 dimensions.')
    if y_pred.ndim != 3:
        raise ValueError('y_pred array must have 3 dimensions.')
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')

    objects_ids = np.unique(y_true[(y_true < 255) & (y_true > 0)])
    nb_objects = len(objects_ids)
    nb_frames = len(y_true)

    jaccard = np.empty((nb_frames, nb_objects), dtype=np.float)

    for i, obj_id in enumerate(objects_ids):
        mask_true, mask_pred = y_true == obj_id, y_pred == obj_id

        union = (mask_true | mask_pred).sum(axis=(1, 2))
        intersection = (mask_true & mask_pred).sum(axis=(1, 2))

        jac = intersection / union
        jac[union == 0.] = 1.
        jaccard[:, i] = jac

    if average_over_objects:
        jaccard = jaccard.mean(axis=1)
    return jaccard
