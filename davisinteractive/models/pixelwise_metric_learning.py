import os

import numpy as np


class PixelWiseMetricLearningModel:
    def __init__(self, embeddings_path):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f'Embeddings not found in {embeddings_path}')
        self.embeddings_path = embeddings_path

    def load_sequence(self, sequence):
        pass

    def __call__(self, points, object_ids):
        pass
