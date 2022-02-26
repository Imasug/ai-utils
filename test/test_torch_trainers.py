import math
import unittest
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from tools.torch.trainers import TorchTrainer


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


test_dir = Path(__file__).parent
root = test_dir.joinpath('data')
checkpoint_dir = test_dir.joinpath('checkpoint')


class MockListener:

    def start(self, target):
        print(f'start name: {target.name}')

    def pre_epoch(self, epoch, target):
        print(f'epoch: {epoch}')

    def post_epoch(self, epoch, data, target):
        print(f'epoch: {epoch}, train loss: {data.train_loss:.3f}, val loss: {data.val_loss:.3f}')

    def end(self):
        print('end')


class TestTorchTrainer(unittest.TestCase):

    def test(self):
        device = torch.device('cpu')

        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=ToTensor(),
        )

        batch_size = 64

        model = NeuralNetwork()

        # ロスはバッチサイズに平均化される。
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def lambda_epoch(epoch):
            return math.pow((1 - epoch / 100), 0.9)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

        listener = MockListener()

        trainer = TorchTrainer(
            name='test',
            epochs=1,
            device=device,
            batch_size=batch_size,
            train_data=training_data,
            val_data=test_data,
            model=model,
            criterion=criterion,
            scheduler=scheduler,
            optimizer=optimizer,
            listener=listener,
            batch_multi=1,
            checkpoint_dir=checkpoint_dir,
        )

        trainer.start()
