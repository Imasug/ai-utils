import unittest
from utils.augmentation import *


class TestAugmentation(unittest.TestCase):

    def test_bi_compose(self):
        def func1(x1, x2):
            return x1 + 1, x2 + 2

        def func2(x1, x2):
            return x1 * 3, x2 * 4

        y = BiCompose([func1, func2])(1, 2)
        self.assertEqual((6, 16), y)

    def test_sync_random_horizontal_flip(self):
        img = Image.fromarray(np.array([[1, 2]]))
        seg = Image.fromarray(np.array([[3, 4]]))
        img, seg = SyncRandomHorizontalFlip()(img, seg)
        # TODO
        print(np.array(img))
        print(np.array(seg))
