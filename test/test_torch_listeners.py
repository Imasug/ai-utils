import unittest
from pathlib import Path

from tools.torch.listeners import TensorBoardLossReporter

log_dir = Path(__file__).parent.joinpath('log')


class TestTorchTensorBoardLossReporter(unittest.TestCase):

    def test(self):
        reporter = TensorBoardLossReporter(log_dir=log_dir)
        reporter.start('test')

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
