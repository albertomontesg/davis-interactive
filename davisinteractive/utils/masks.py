import numpy as np


def combine_masks(masks, th=0.5, method='max_per_pixel'):
    """ Combine mask for different objects

    # Arguments
        masks: List containing a list of masks for every object. Therefore, len(masks) == number_objects
               and len(masks[0]) == number_frames. The masks should be numpy array.
        th: Threshold to binarize the masks. Default: 0.5
        method: Method that specifies how the masks are fuse, the following are available:
                - max_per_pixel(default): Computes

    # Returns
        list: Returns a list with all the results of the masks fused
    """
    n_frames = len(masks[0])
    n_objects = len(masks)
    output_masks = []
    for fr_id in range(n_frames):
        res_mat = np.zeros((masks[0].shape[0], masks[0].shape[1], n_objects))
        for obj_id in range(n_objects):
            res_mat[:, :, obj_id] = masks[obj_id]
        marker = np.argmax(res_mat, axis=2)
        for obj_id in range(n_objects):
            output_masks.append(np.logical_and(marker == obj_id, masks[obj_id] > th))

    return output_masks
