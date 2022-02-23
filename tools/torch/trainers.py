from tqdm import tqdm
import torch


class TorchTrainer:

    def __init__(
            self,
            epochs,
            device,
            train_data,
            val_data,
            model,
            criterion,
            optimizer,
            callback,
    ):
        self.epochs = epochs
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.callback = callback

        self.model.to(self.device)

    def train(self, epoch):
        self.model.train()
        bar = tqdm(self.train_data)
        train_loss = 0.0
        for i, (data, target) in enumerate(bar, start=1):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
            self.optimizer.step()

            train_loss += loss

            bar.set_description(f'epoch: {epoch}, iter: {i}, loss: {loss}')

        return train_loss / len(bar)

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

            bar.set_description(f'epoch: {epoch}, iter: {i}, loss: {loss}')

        return val_loss / len(bar)

    def start(self):
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + self.epochs):
            train_loss = self.train(epoch)
            val_loss = self.val(epoch)
            self.callback(epoch, train_loss, val_loss, self)
