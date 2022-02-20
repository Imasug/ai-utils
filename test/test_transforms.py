import unittest
from utils.transforms import *
from pathlib import Path
import matplotlib.pyplot as plt

test_images_path = Path(__file__).parent.joinpath('images')
test_img_path = test_images_path.joinpath('test.jpg')
test_seg_path = test_images_path.joinpath('test.png')


class TestAugmentation(unittest.TestCase):

    def test_transforms(self):
        def transform1(img, seg):
            return img + 1, seg + 2

        def transform2(img, seg):
            return img * 3, seg * 4

        y = Transforms([transform1, transform2])(img=1, seg=2)
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

    def test_transform(self):
        img = 1
        seg = 2

        img, seg = Transform(img=lambda x: 2 * x, seg=lambda x: 3 * x)(img, seg)

        self.assertEqual(2, img)
        self.assertEqual(6, seg)

    def test_sync_resize(self):
        row = 2
        col = 2
        fig = plt.figure()

        img = Image.open(test_img_path)
        seg = Image.open(test_seg_path)

        fig.add_subplot(row, col, 1)
        plt.imshow(img)
        fig.add_subplot(row, col, 2)
        plt.imshow(seg)

        img, seg = SyncResize(size=(200, 200))(img, seg)

        fig.add_subplot(row, col, 3)
        plt.imshow(img)
        fig.add_subplot(row, col, 4)
        plt.imshow(seg)
        plt.show()

    def test_random_gaussian_blur(self):
        row = 2
        col = 2
        fig = plt.figure()

        img = Image.open(test_img_path)
        seg = Image.open(test_seg_path)

        fig.add_subplot(row, col, 1)
        plt.imshow(img)
        fig.add_subplot(row, col, 2)
        plt.imshow(seg)

        img = RandomGaussianBlur()(img)

        fig.add_subplot(row, col, 3)
        plt.imshow(img)
        fig.add_subplot(row, col, 4)
        plt.imshow(seg)
        plt.show()
