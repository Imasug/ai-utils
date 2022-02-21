import unittest

import mxnet as mx

import utils.functional as F


class TestFunctional(unittest.TestCase):

    def test_calc_mean_std(self):
        items = []
        data = mx.nd.array([
            # R
            [
                [1, 2],
                [3, 4],
            ],
            # G
            [
                [2, 4],
                [6, 8],
            ],
            # B
            [
                [3, 6],
                [9, 12],
            ]
        ])

        items.append((data, 1))
        items.append((data, 2))

        dataset = mx.gluon.data.ArrayDataset(items)
        mean, std = F.calc_mean_std(dataset)

        self.assertEqual([2.5, 5, 7.5], mean)
        self.assertEqual([1.118, 2.236, 3.354], std)
