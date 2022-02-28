import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

LATEST_FILENAME = 'checkpoint'
MODEL_FILENAME = 'model.pth'
OPTIM_FILENAME = 'optim.pth'
SCHED_FILENAME = 'sched.pth'
EPOCH_FOLDER_TEMPLATE = 'epoch_%s'


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
            scheduler,
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
        self.scheduler = scheduler
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
                loss = self.criterion(output, target.long())

            val_loss += loss

            bar.set_description(f'epoch: {epoch}, step: {i}, loss: {loss:.3f}')

        return val_loss / len(bar)

    def load(self):
        with open(self.latest_file) as f:
            data = json.load(f)
        epoch = data['epoch']

        folder = self.checkpoint_dir.joinpath(EPOCH_FOLDER_TEMPLATE % epoch)

        # model
        model_file = folder.joinpath(MODEL_FILENAME)
        self.model.load_state_dict(torch.load(model_file))

        # optimizer
        optim_file = folder.joinpath(OPTIM_FILENAME)
        self.optimizer.load_state_dict(torch.load(optim_file))

        # scheduler
        sched_file = folder.joinpath(SCHED_FILENAME)
        self.scheduler.load_state_dict(torch.load(sched_file))

        return epoch + 1

    def save(self, epoch):

        folder = self.checkpoint_dir.joinpath(EPOCH_FOLDER_TEMPLATE % epoch)
        folder.mkdir(exist_ok=True)

        # model
        model_file = folder.joinpath(MODEL_FILENAME)
        torch.save(self.model.state_dict(), model_file)

        # optimizer
        # TODO
        optim_file = folder.joinpath(OPTIM_FILENAME)
        torch.save(self.optimizer.state_dict(), optim_file)

        # scheduler
        # TODO
        sched_file = folder.joinpath(SCHED_FILENAME)
        torch.save(self.scheduler.state_dict(), sched_file)

        latest = {
            'epoch': epoch,
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
                self.scheduler.step()
                self.save(epoch)
                data = type("data", (object,), {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                })
                self.listener.post_epoch(epoch, data, self)
        finally:
            self.listener.end()
