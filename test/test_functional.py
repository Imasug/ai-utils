import unittest

import numpy as np
import torch

import utils.functional as F
from PIL import Image


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

        mean, std = F.calc_img_mean_std(data)

        self.assertEqual([2.5, 5, 7.5], [v.item() for v in mean * 255])
        self.assertEqual([1.118, 2.236, 3.354], [round(v.item(), 3) for v in std * 255])

    def test_get_seg_metrics(self):
        target = np.array([
            # 1
            [
                [
                    [1, 2],
                    [1, 2],
                ]
            ],
            # 2
            [
                [
                    [1, 2],
                    [1, 2],
                ]
            ],
        ])
        prediction = np.array([
            # 1
            [
                [
                    [2, 1],
                    [1, 2],
                ]
            ],
            # 2
            [
                [
                    [2, 1],
                    [1, 2],
                ]
            ],
        ])

        cls_num = 5

        seg_metrics = F.get_seg_metrics(target, prediction, cls_num)

        self.assertEqual(0.5, seg_metrics.get_pixel_accuracy())
        self.assertEqual(0.5, seg_metrics.get_mean_accuracy())
        self.assertEqual(1 / 3, seg_metrics.get_mean_iou())

        # torch
        target = torch.from_numpy(target)
        prediction = torch.from_numpy(prediction)

        seg_metrics = F.get_seg_metrics(target, prediction, cls_num)

        self.assertEqual(0.5, seg_metrics.get_pixel_accuracy())
        self.assertEqual(0.5, seg_metrics.get_mean_accuracy())
        self.assertEqual(1 / 3, seg_metrics.get_mean_iou())
