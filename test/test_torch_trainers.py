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

        batch_size = 4

        model = NeuralNetwork()

        # ロスはバッチサイズに平均化される。
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def callback(epoch, train_loss, val_loss, this):
            print(f'epoch: {epoch}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}')

        trainer = TorchTrainer(
            epochs=5,
            device=device,
            batch_size=batch_size,
            train_data=training_data,
            val_data=test_data,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callback=callback,
            batch_multi=16,
        )

        trainer.start()
