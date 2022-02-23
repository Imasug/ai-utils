import unittest

import numpy as np

import utils.functional as F
from PIL import Image
import mxnet as mx


class TestFunctional(unittest.TestCase):

    def test_calc_img_mean_std(self):
        img = np.array([
            [[1, 2, 3], [2, 4, 6]],
            [[3, 6, 9], [4, 8, 12]],
        ]).astype(np.uint8)
        img = Image.fromarray(img)

        data = []
        data.append((img, 'label'))
        data.append((img, 'label'))

        dataset = mx.gluon.data.ArrayDataset(data)

        mean, std = F.calc_img_mean_std(dataset)

        self.assertEqual([2.5, 5, 7.5], [v.item() for v in mean * 255])
        self.assertEqual([1.118, 2.236, 3.354], [round(v.item(), 3) for v in std * 255])
