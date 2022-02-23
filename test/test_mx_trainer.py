import unittest
from pathlib import Path

import mxnet as mx
from mxnet import gluon

from tools.mx.trainer import Trainer


class TestTrainer(unittest.TestCase):

    def test(self):
        model = mx.gluon.nn.Dense(1)
        model.initialize()

        batch_multi = 2

        data = mx.nd.array([
            1, 2, 3
        ])
        target = mx.nd.array([
            0, 1, 0
        ])
        items = [
            (data, target),
            (data, target),
            (data, target),
        ]
        dataset = mx.gluon.data.ArrayDataset(items)

        criterion = gluon.loss.SoftmaxCrossEntropyLoss()

        class MockOptimizer:
            def step(self, batch_size):
                pass

        optimizer = MockOptimizer()

        def callback(epoch, train_loss, val_loss, this):
            print(f'epoch: {epoch}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}')

        trainer = Trainer(
            model=model,
            epochs=1,
            train_data=dataset,
            val_data=dataset,
            criterion=criterion,
            optimizer=optimizer,
            callback=callback,
            save_dir=str(Path(__file__).parent.joinpath('checkpoints')),
            batch_multi=batch_multi,
        )

        trainer.start()
