import unittest

import torch
from torch import nn

from tools.torch.listeners import *

log_dir = Path(__file__).parent.joinpath('log')


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
