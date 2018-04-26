import os
import unittest

import numpy as np
from PIL import Image

from .mask import combine_masks

TEST_DIR = os.path.join(os.path.dirname(__file__), 'masks_test')


class TestCombineMasks(unittest.TestCase):

    def test_combine_masks(self):
        n_objects = 2
        n_frames = 3
        all_masks = []
        for obj_id in range(n_objects):
            obj_masks = []
            for fr_id in range(n_frames):
                obj_masks.append(
                    np.array(
                        Image.open(
                            os.path.join(TEST_DIR, 'input_masks',
                                         str(obj_id + 1),
                                         '{:05d}.png'.format(fr_id)))))
            all_masks.append(obj_masks)
        final_mask = combine_masks(all_masks)
        for fr_id in range(n_frames):
            gt_mask = np.array(
                Image.open(
                    os.path.join(TEST_DIR, 'output_masks',
                                 '{:05d}.png'.format(fr_id))))
            assert (final_mask[fr_id] == gt_mask).all()
