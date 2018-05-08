""" Interface for manipulating masks stored in RLE format.
"""
import numpy as np

from davisinteractive.third_party.mask_api import _mask

__all__ = [
    'encode_mask', 'decode_mask', 'encode_batch_masks', 'decode_batch_masks'
]


def encode_mask(mask, nb_objects=None):
    """ Encode a mask.

    It accepts multiple indexes on the mask. The mask for every index will be
    encoded in a different RLE.

    # Arguments
        bimask: Numpy Array. Mask array with the index of every pixel. The shape
            must be (H, W) and all the values are supposed to be integers.
        nb_objects: Integer. Number of objects in the mask. If not given the
            value will be infered. If given, the computation will be faster.

    # Return
        Dictionary: Dictionary with the RLE of the mask.
    """
    mask = mask.astype(np.int)
    assert mask.ndim == 2
    h, w = mask.shape

    if nb_objects is None:
        obj_ids = np.unique(mask[mask > 0])
    else:
        obj_ids = np.arange(nb_objects, dtype=np.uint8) + 1
    obj_ids = obj_ids.reshape(1, 1, -1)

    binmask = mask.reshape(h, w, 1) == obj_ids
    binmask = np.asfortranarray(binmask).astype(np.uint8)

    rle_objs = _mask.encode(binmask)
    encoding = {'size': [h, w], 'objects': []}
    for i, obj_id in enumerate(obj_ids.ravel()):
        encoding['objects'].append({
            'object_id': int(obj_id),
            'counts': rle_objs[i]['counts'].decode()
        })

    return encoding


def decode_mask(encoding):
    """ Decode a mask.

    Decode a multi index mask and return its mask as a Numpy Array.

    # Arguments
        encoding: Dictionary. Mask encoded object.

    # Returns
        Numpy Array: Mask decoded with shape (H, W).
    """
    h, w = encoding['size']
    obj_ids = [o['object_id'] for o in encoding['objects']]
    rle_objs = [{
        'size': [h, w],
        'counts': o['counts'].encode()
    } for o in encoding['objects']]
    binmask = _mask.decode(rle_objs)
    obj_ids = np.asarray(obj_ids, dtype=binmask.dtype).reshape(1, 1, -1)
    mask = (binmask * obj_ids).max(axis=2)

    return mask


def encode_batch_masks(masks, nb_objects=None):
    """ Encode a batch of masks.

    It accepts multiple indexes on the mask. The mask for every index will be
    encoded in a different RLE.

    # Arguments
        bimask: Numpy Array. Mask array with the index of every pixel. The shape
            must be (B, H, W) and all the values are supposed to be integers.
        nb_objects: Integer. Number of objects in the mask. If not given the
            value will be infered. If given, the computation will be faster.

    # Return
        Dictionary: Dictionary with the RLE of the mask.
    """
    assert masks.ndim == 3
    b, h, w = masks.shape
    encoding = {'size': [b, h, w], 'frames': []}
    for i in range(b):
        enc = encode_mask(masks[i], nb_objects=nb_objects)
        encoding['frames'].append(enc['objects'])

    return encoding


def decode_batch_masks(encoding):
    """ Decode a batch of mask.

    Decode a multi index mask and return its mask as a Numpy Array.

    # Arguments
        encoding: Dictionary. Mask encoded object.

    # Returns
        Numpy Array: Mask decoded with shape (B, H, W).
    """
    b, h, w = encoding['size']
    frames_obj = [{'size': [h, w], 'objects': o} for o in encoding['frames']]
    masks = [decode_mask(obj) for obj in frames_obj]
    masks = np.stack(masks, axis=0)
    assert masks.shape == (b, h, w)

    return masks
