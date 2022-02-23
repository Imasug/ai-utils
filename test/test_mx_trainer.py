import unittest
from pathlib import Path

import mxnet as mx
from mxnet import gluon

from tools.mx.trainer import Trainer
from gluoncv.loss import MixSoftmaxCrossEntropyLoss
from gluoncv.utils.parallel import *


class TestTrainer(unittest.TestCase):

    def test(self):
        ctx_list = [mx.cpu(0)]

        class MockModel(gluon.nn.Block):

            def __init__(self):
                super(MockModel, self).__init__()
                self.proc1 = gluon.nn.Dense(1)
                self.proc2 = gluon.nn.Dense(1)

            def forward(self, x):
                return self.proc1(x), self.proc2(x)

        model = MockModel()
        model.initialize()
        model = DataParallelModel(model, ctx_list)

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

        criterion = MixSoftmaxCrossEntropyLoss(aux=True)
        criterion = DataParallelCriterion(criterion, ctx_list)

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
