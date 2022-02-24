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
        })

        reporter = TorchTensorBoardModelReporter(log_dir=log_dir)
        reporter.start(target)
