from torch import nn
import torch.nn.init as initer


def init_weights(m):
    if isinstance(m, nn.modules.conv._ConvNd):
        initer.xavier_normal_(m.weight)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        initer.normal_(m.weight, 1.0, 0.02)


PALETTE = [
    0, 0, 0,
    64, 0, 0,
    0, 64, 0,
    0, 0, 64,
    64, 64, 0,
    64, 0, 64,
    0, 64, 64,
    64, 64, 64,
    128, 0, 0,
    0, 128, 0,
    0, 0, 128,
    128, 128, 0,
    128, 0, 128,
    0, 128, 128,
    128, 128, 128,
    192, 0, 0,
    0, 192, 0,
    0, 0, 192,
    192, 192, 0,
    192, 0, 192,
    0, 192, 192,
    192, 192, 192,
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 0,
    255, 0, 255,
    0, 255, 255,
]
