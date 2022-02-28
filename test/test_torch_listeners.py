import random
import unittest

import torch
from torch import nn

from tools.torch.listeners import *
from PIL import Image
from torchvision.transforms import ToTensor
from utils.transforms import Transform

log_dir = Path(__file__).parent.joinpath('log')
test_images_path = Path(__file__).parent.joinpath('images')
test_img_path = test_images_path.joinpath('test.jpg')
test_seg_path = test_images_path.joinpath('test.png')


class TestTorchTensorBoardLossReporter(unittest.TestCase):

    def test(self):
        target = type('target', (object,), {
            'name': 'test'
        })

        reporter = TensorBoardLossReporter(log_dir=log_dir)
        reporter.start(target)

        train_loss = val_loss = 100

        for i in range(1, 101):
            data = type('data', (object,), {
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            train_loss -= 5 / i
            val_loss -= 4 / i
            reporter.post_epoch(i, data, None)

        reporter.end()


class TestTorchTensorBoardModelReporter(unittest.TestCase):

    def test(self):
        model = nn.Linear(3, 3)

        dataset = [
            (torch.tensor([1., 1., 1.]), 1)
        ]

        val_data = type('val_data', (object,), {
            'dataset': dataset
        })

        target = type('target', (object,), {
            'name': 'test',
            'model': model,
            'val_data': val_data,
            'device': 'cpu'
        })

        reporter = TensorBoardModelReporter(log_dir=log_dir)
        reporter.start(target)


class TestListeners(unittest.TestCase):

    def test(self):
        class MockListener(Listener):

            def start(self, target):
                print('start')

            def pre_epoch(self, epoch, target):
                print('pre_epoch')

            def post_epoch(self, epoch, data, target):
                print('post_epoch')

            def end(self):
                print('end')

        listeners = Listeners([
            MockListener(),
            MockListener(),
        ])

        listeners.start(None)
        listeners.pre_epoch(1, None)
        listeners.post_epoch(1, None, None)
        listeners.end()


class TestPostEpochCopier(unittest.TestCase):

    # TODO ファイルコピー
    # TODO テストケース不足

    def test_copy_dir(self):
        src_path = Path('src')
        src_path.mkdir()
        dst_path = Path('dst')
        listener = PostEpochCopier(src_path, dst_path)
        listener.post_epoch(None, None, None)
        self.assertTrue(dst_path.exists())
        src_path.rmdir()
        dst_path.rmdir()


class TestTensorBoardSegmentationInferenceReporter(unittest.TestCase):

    def test(self):
        img = Image.open(test_img_path)
        seg = Image.open(test_seg_path)

        def model(x):
            return None

        target = type('target', (object,), {
            'name': 'test',
            'model': model,
            'device': 'cpu'
        })

        def inference(model, img):
            r = int(random.random() * 255)
            return Image.new('P', (100, 100), (r, 0, 0))

        reporter = TensorBoardSegmentationInferenceReporter(
            log_dir=log_dir,
            dataset=[(img, seg)],
            inference=inference,
        )

        reporter.start(target)

        for i in range(1, 4):
            reporter.post_epoch(i, None, target)

        reporter.end()
