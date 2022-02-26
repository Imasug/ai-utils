from turtle import forward
from torch import nn


class AuxiliaryLoss(nn.Module):

    def __init__(self, aux_weight=0.4) -> None:
        super(AuxiliaryLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, output, target):
        x, aux = output
        main_loss = nn.CrossEntropyLoss()(x, target)
        aux_loss = nn.CrossEntropyLoss()(aux, target)
        return main_loss + aux_loss * self.aux_weight
