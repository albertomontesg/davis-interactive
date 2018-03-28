import os

import h5py
import numpy as np


class PixelWiseMetricLearningModel:
    IMG_SIZE = (60, 120)

    def __init__(self, embeddings_path):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f'Embeddings not found in {embeddings_path}')
        self.embeddings_path = embeddings_path

        self.embeddings = None

    def load_sequence(self, sequence):

        dataset = h5py.File(self.embeddings_path, 'r')
        if sequence not in dataset:
            dataset.close()
            raise ValueError(
                f'Sequence not in dataset at: {self.embeddings_path}')

        self.embeddings = dataset[sequence][...]
        dataset.close()

    def __call__(self, points, object_ids):
        E = self.embeddings

        f, h, w, d = E.shape

        E = E.reshape(f * h * w, d)

        P = np.ravel_multi_index(points.T, (f, h, w))
        X = E[P, :]

        S = E.dot(X.T)
        S_closest = S.argmax(axis=1)
        M = object_ids[S_closest]
        M = M.reshape(f, h, w)
        return M
