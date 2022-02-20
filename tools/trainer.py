import json
import os

import numpy as np
from mxnet import autograd
from tqdm import tqdm


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
            save_dir,
            batch_multi=1,
    ):
        self.model = model
        self.epochs = epochs
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.callback = callback
        self.save_dir = save_dir
        self.batch_multi = batch_multi

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
        return val_loss / len(self.val_data)

    def start(self):
        latest_file = f'{self.save_dir}/latest.json'
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                latest = json.load(f)
            params_file = latest["params_file"]
            self.model.load_params(f'{self.save_dir}/{params_file}')
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train()
            val_loss = self.validate()
            self.callback(train_loss, val_loss)
            params_file = f'epoch_{epoch}.params'
            self.model.save_parameters(f'{self.save_dir}/{params_file}')
            latest = {
                'params_file': params_file,
            }
            with open(latest_file, 'w') as f:
                json.dump(latest, f, indent=2)
