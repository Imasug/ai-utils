import unittest
from utils.augmentation import *
from pathlib import Path
import matplotlib.pyplot as plt

test_images_path = Path(__file__).parent.joinpath('images')
test_img_path = test_images_path.joinpath('test.jpg')
test_seg_path = test_images_path.joinpath('test.png')


class TestAugmentation(unittest.TestCase):

    def test_bi_compose(self):
        def func1(x1, x2):
            return x1 + 1, x2 + 2

        def func2(x1, x2):
            return x1 * 3, x2 * 4

        y = BiCompose([func1, func2])(1, 2)
        self.assertEqual((6, 16), y)

    def test_sync_random_horizontal_flip(self):
        row = 2
        col = 2
        fig = plt.figure()

        img = Image.open(test_img_path)
        seg = Image.open(test_seg_path)

        fig.add_subplot(row, col, 1)
        plt.imshow(img)
        fig.add_subplot(row, col, 2)
        plt.imshow(seg)

        img, seg = SyncRandomHorizontalFlip()(img, seg)

        fig.add_subplot(row, col, 3)
        plt.imshow(img)
        fig.add_subplot(row, col, 4)
        plt.imshow(seg)
        plt.show()

    def test_sync_random_rotation(self):
        row = 2
        col = 2
        fig = plt.figure()

        img = Image.open(test_img_path)
        seg = Image.open(test_seg_path)

        fig.add_subplot(row, col, 1)
        plt.imshow(img)
        fig.add_subplot(row, col, 2)
        plt.imshow(seg)

        img, seg = SyncRandomRotation((-45, 45))(img, seg)

        fig.add_subplot(row, col, 3)
        plt.imshow(img)
        fig.add_subplot(row, col, 4)
        plt.imshow(seg)
        plt.show()

    def test_sync_random_scale(self):
        row = 2
        col = 2
        fig = plt.figure()

        img = Image.open(test_img_path)
        seg = Image.open(test_seg_path)

        fig.add_subplot(row, col, 1)
        plt.imshow(img)
        fig.add_subplot(row, col, 2)
        plt.imshow(seg)

        img, seg = SyncRandomScaledCrop((0.5, 2.0))(img, seg)

        fig.add_subplot(row, col, 3)
        plt.imshow(img)
        fig.add_subplot(row, col, 4)
        plt.imshow(seg)
        plt.show()
