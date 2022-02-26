import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

LATEST_FILENAME = 'checkpoint'


class TorchTrainer:

    def __init__(
            self,
            name,
            epochs,
            device,
            batch_size,
            train_data,
            val_data,
            model,
            criterion,
            optimizer,
            listener,
            checkpoint_dir: Path,
            num_workers=0,
            pin_memory=False,
            batch_multi=1,
    ):
        self.name = name
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.train_data = DataLoader(train_data, shuffle=True, drop_last=True, batch_size=self.batch_size,
                                     num_workers=num_workers, pin_memory=pin_memory)
        self.val_data = DataLoader(val_data, shuffle=True, drop_last=True, batch_size=self.batch_size,
                                   num_workers=num_workers, pin_memory=pin_memory)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.listener = listener
        self.batch_multi = batch_multi
        self.checkpoint_dir = checkpoint_dir.joinpath(self.name)
        self.latest_file = self.checkpoint_dir.joinpath(LATEST_FILENAME)

        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

    def train(self, epoch):
        self.model.train()
        bar = tqdm(self.train_data)
        train_loss = 0.0
        multi = 0
        step = 0
        for i, (data, target) in enumerate(bar, start=1):
            data = data.to(self.device)
            target = target.to(self.device)

            if multi == 0:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = self.model(data)
                loss = self.criterion(output, target.long())
                adjusted_loss = loss / self.batch_multi
                adjusted_loss.backward()
            multi += 1

            if multi == self.batch_multi:
                self.optimizer.step()
                step += 1
                multi = 0
                train_loss += loss
                bar.set_description(f'epoch: {epoch}, step: {step}, loss: {loss:.3f}')

        return train_loss / step

    def val(self, epoch):
        self.model.eval()
        bar = tqdm(self.val_data)
        val_loss = 0.0
        for i, (data, target) in enumerate(bar, start=1):
            data = data.to(self.device)
            target = target.to(self.device)

            with torch.set_grad_enabled(False):
                output = self.model(data)
                loss = self.criterion(output, target)

            val_loss += loss

            bar.set_description(f'epoch: {epoch}, step: {i}, loss: {loss:.3f}')

        return val_loss / len(bar)

    def load(self):
        # TODO ファイル名からエポック数を抜き出した方がよさそう
        with open(self.latest_file) as f:
            data = json.load(f)
        pth_file = self.checkpoint_dir.joinpath(data['path'])
        state = torch.load(pth_file)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        return data['epoch'] + 1

    def save(self, epoch):
        pth_filename = f'epoch_{epoch}.pth'
        pth_file = self.checkpoint_dir.joinpath(pth_filename)
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, pth_file)
        latest = {
            'epoch': epoch,
            'path': pth_filename
        }
        with open(self.latest_file, 'w') as f:
            json.dump(latest, f, indent=2)

    def start(self):
        self.listener.start(self)
        try:
            start_epoch = self.load() if self.latest_file.exists() else 1
            for epoch in range(start_epoch, start_epoch + self.epochs):
                self.listener.pre_epoch(epoch, self)
                train_loss = self.train(epoch)
                val_loss = self.val(epoch)
                self.save(epoch)
                data = type("data", (object,), {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                })
                self.listener.post_epoch(epoch, data, self)
        finally:
            self.listener.end()
