from mxnet import gluon, autograd
from tqdm import tqdm
import numpy as np


class Trainer:

    def __init__(
            self,
            model,
            epochs,
            train_data,
            val_data,
            criterion,
            optimizer,
            callback,
            batch_multi=1,
    ):
        self.model = model
        self.epochs = epochs
        self.batch_multi = batch_multi
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.callback = callback

        for p in self.model.collect_params().values():
            p.grad_req = 'add'

    def train(self):
        bar = tqdm(self.train_data)
        train_loss = 0.0
        batch_size = 0
        multi = 0
        i = 0
        for data, target in bar:
            with autograd.record():
                outputs = self.model(data)
                losses = self.criterion(*outputs, target)
                autograd.backward(losses)
            batch_size += data.shape[0]
            multi += 1
            if multi == self.batch_multi:
                self.optimizer.step(batch_size)
                for p in self.model.collect_params().values():
                    p.zero_grad()
                batch_size = 0
                multi = 0
                i += 1
                for loss in losses:
                    train_loss += np.mean(loss.asnumpy()) / len(losses)
                bar.set_description(f'iter: {i}, train loss: {train_loss / i:.3f}')
                # TODO 消す
                break
        return train_loss / i

    def validate(self):
        bar = tqdm(self.val_data)
        val_loss = 0.0
        for i, (data, target) in enumerate(bar, start=1):
            outputs = self.model(data)
            losses = self.criterion(*outputs, target)
            for loss in losses:
                val_loss += np.mean(loss.asnumpy()) / len(losses)
            bar.set_description(f'iter: {i}, val loss: {val_loss / i:.3f}')
            # TODO 消す
            break
        return val_loss / len(self.val_data)

    def start(self):
        # TODO セーブデータ取得
        for epoch in range(0, self.epochs):
            train_loss = self.train()
            val_loss = self.validate()
            self.callback(train_loss, val_loss)
            # TODO セーブ
