from __future__ import absolute_import

import json
import unittest

import numpy as np
from PIL import Image

from davisinteractive.common import Path
from davisinteractive.third_party import mask_api


class TestSingleMaskEncoding(unittest.TestCase):

    test_mask = None

    @classmethod
    def setUpClass(cls):
        img_path = Path(__file__).parent / 'test_files' / '00000.png'
        mask = np.asarray(Image.open(img_path))
        cls.test_mask = mask

    def test(self):
        encoded_object = mask_api.encode_mask(self.test_mask)
        decoded_mask = mask_api.decode_mask(encoded_object)

        assert self.test_mask.dtype == decoded_mask.dtype
        assert self.test_mask.shape == decoded_mask.shape
        assert np.all(self.test_mask == decoded_mask)


class TestMultiMaskEncoding(unittest.TestCase):

    test_masks = None

    @classmethod
    def setUpClass(cls):
        img_dir = Path(__file__).parent / 'test_files'
        masks = []
        for i in range(4):
            img_path = img_dir / '{:05}.png'.format(i)
            masks.append(np.asarray(Image.open(img_path)))

        masks = np.stack(masks, axis=0)
        assert masks.shape[0] == 4
        cls.test_masks = masks

    def test(self):
        encoded_object = mask_api.encode_batch_masks(self.test_masks)
        decoded_masks = mask_api.decode_batch_masks(encoded_object)

        assert self.test_masks.dtype == decoded_masks.dtype
        assert self.test_masks.shape == decoded_masks.shape
        assert np.all(self.test_masks == decoded_masks)

    def test_json_encoding(self):
        encoded_object = mask_api.encode_batch_masks(self.test_masks)
        json_encoded_object = json.JSONEncoder().encode(encoded_object)
        encoded_object = json.JSONDecoder().decode(json_encoded_object)
        decoded_masks = mask_api.decode_batch_masks(encoded_object)

        assert self.test_masks.dtype == decoded_masks.dtype
        assert self.test_masks.shape == decoded_masks.shape
        assert np.all(self.test_masks == decoded_masks)
