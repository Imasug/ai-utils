import unittest
from datasets.generic_dataset import GenericSegmentationDataset, COCODataset
from pathlib import Path
from PIL import Image


class TestGenericSegmentationDataset(unittest.TestCase):

    def test_train(self):
        root = Path(__file__).parent.joinpath('datasets').joinpath('lip')
        dataset = GenericSegmentationDataset(root, 'train')
        self.assertEqual(2, len(dataset))

        img0, seg0 = dataset.__getitem__(0)
        self.assertIsInstance(img0, Image.Image)
        self.assertEqual('RGB', img0.mode)
        self.assertIsInstance(seg0, Image.Image)
        self.assertEqual('L', seg0.mode)
        self.assertEqual((179, 312), img0.size)
        self.assertEqual(img0.size, seg0.size)

        img1, seg1 = dataset.__getitem__(1)
        self.assertEqual((104, 120), img1.size)
        self.assertEqual(img1.size, seg1.size)

    def test_val(self):
        root = Path(__file__).parent.joinpath('datasets').joinpath('lip')
        dataset = GenericSegmentationDataset(root, 'val')
        self.assertEqual(1, len(dataset))

    def test_transform(self):
        root = Path(__file__).parent.joinpath('datasets').joinpath('lip')

        def transform(data, target):
            return 1, 2

        dataset = GenericSegmentationDataset(root, 'val', transform=transform)
        img, seg = dataset.__getitem__(0)
        self.assertEqual(img, 1)
        self.assertEqual(seg, 2)


class TestCOCODataset(unittest.TestCase):

    def test(self):
        root = Path(__file__).parent.joinpath('datasets').joinpath('fashionpedia')
        dataset = COCODataset(root=root, mode='train')
        self.assertEqual(1158, len(dataset))
        img0, seg0 = dataset.__getitem__(0)

        self.assertIsInstance(img0, Image.Image)
        self.assertEqual('RGB', img0.mode)
        self.assertEqual((771, 1024), img0.size)

        self.assertIsInstance(seg0, Image.Image)
        self.assertEqual('L', seg0.mode)
        self.assertEqual((771, 1024), seg0.size)

    def test_transform(self):
        root = Path(__file__).parent.joinpath('datasets').joinpath('fashionpedia')

        def transform(data, target):
            return 1, 2

        dataset = COCODataset(root=root, mode='train', transform=transform)

        img, seg = dataset.__getitem__(0)
        self.assertEqual(img, 1)
        self.assertEqual(seg, 2)

    # TODO test conv_cat
