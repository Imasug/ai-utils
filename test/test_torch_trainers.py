import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
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


root = Path(__file__).parent.joinpath('data')


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

        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        model = NeuralNetwork()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def callback(epoch, train_loss, val_loss, this):
            print(f'epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}')

        trainer = TorchTrainer(
            epochs=5,
            device=device,
            train_data=train_dataloader,
            val_data=test_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callback=callback,
        )

        trainer.start()
