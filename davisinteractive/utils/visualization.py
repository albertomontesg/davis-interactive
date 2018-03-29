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
