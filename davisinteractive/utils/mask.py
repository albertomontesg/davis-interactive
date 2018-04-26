import numpy as np


def combine_masks(masks, th=0.5, method='max_per_pixel'):
    """ Combine mask for different objects.

    Different methods are the following:

    * `max_per_pixel`: Computes the final mask taking the pixel with the highest
                       probability for every object.

    # Arguments
        masks: List. Containing a list of masks for every object.
            Therefore, `len(masks) == number_objects` and
            `len(masks[0]) == number_frames`. The masks should be Numpy Array.
        th: Float. Threshold to binarize the masks.
        method: String. Method that specifies how the masks are fused.

    # Returns
        list: Returns a list with all the results of the masks fused.
    """
    n_frames = len(masks[0])
    n_objects = len(masks)
    output_masks = np.zeros((n_frames, masks[0][0].shape[0],
                            masks[0][0].shape[1]))
    for fr_id in range(n_frames):
        res_mat = np.zeros((masks[0][fr_id].shape[0], masks[0][fr_id].shape[1],
                            n_objects))
        for obj_id in range(n_objects):
            res_mat[:, :, obj_id] = masks[obj_id][fr_id]
        marker = np.argmax(res_mat, axis=2)
        out_mask = np.zeros((masks[0][fr_id].shape[0],
                            masks[0][fr_id].shape[1]))
        for obj_id in range(n_objects):
            tmp_mask = np.logical_and(marker == obj_id,
                                      masks[obj_id][fr_id] > th)
            out_mask[tmp_mask] = obj_id + 1
        output_masks[fr_id, :, :] = out_mask
    return output_masks
