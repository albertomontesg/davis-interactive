from __future__ import absolute_import, division

import numpy as np


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def plot_scribble(ax, scribble, frame, output_size=None, **kwargs):
    """ Plot scribbles into an axis.

    # Arguments
        ax: Matplotlib Axis. Axis where to plot the scribble lines.
        scribbles: Scribble. Scribble to plot.
        frame: Integer. Frame of the scribble to plot.
        output_size: Tuple. Image size to scale the scribble points `(H, W)`.
        **kwargs: Dictionary. Additional parameters to pass at the
            `ax.plot(**kwargs)` method.

    # Returns
        matplotlib.axis: Returns the given axis with the scribbles plotted on
            it.
    """
    scribbles = scribble['scribbles']
    if frame >= len(scribbles):
        raise ValueError('Frame value not valid')

    cmap = _pascal_color_map(normalized=True)

    frame_scribbles = scribbles[frame]

    for line in frame_scribbles:
        path, obj_id = line['path'], line['object_id']
        path = np.asarray(path, dtype=np.float32)
        color = cmap[obj_id]

        if output_size:
            img_size = np.asarray(output_size, dtype=np.float32)
            img_size -= 1
            path *= img_size

        ax.plot(*path.T, color=color, **kwargs)

    return ax


def overlay_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    """ Overlay mask over image.

    This function allows you to overlay a mask over an image with some
    transparency.

    # Arguments
        im: Numpy Array. Array with the image. The shape must be (H, W, 3) and
            the pixels must be represented as `np.uint8` data type.
        ann: Numpy Array. Array with the mask. The shape must be (H, W) and the
            values must be intergers
        alpha: Float. Proportion of alpha to apply at the overlaid mask.
        colors: Numpy Array. Optional custom colormap. It must have shape (N, 3)
            being N the maximum number of colors to represent.
        contour_thickness: Integer. Thickness of each object index contour draw
            over the overlay. This function requires to have installed the
            package `opencv-python`.

    # Returns
        Numpy Array: Image of the overlay with shape (H, W, 3) and data type
            `np.uint8`.
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img
