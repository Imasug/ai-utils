import unittest
from PIL import Image
import numpy as np
from utils.augmentation import SyncRandomHorizontalFlip


class TestAugmentation(unittest.TestCase):

    def test_sync_random_horizontal_flip(self):
        img = Image.fromarray(np.array([[1, 2]]))
        seg = Image.fromarray(np.array([[3, 4]]))
        img, seg = SyncRandomHorizontalFlip()(img, seg)
        print(np.array(img))
        print(np.array(seg))
