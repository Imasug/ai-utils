from torch import nn
import torch.nn.init as initer


def init_weights(m):
    if isinstance(m, nn.modules.conv._ConvNd):
        initer.xavier_normal_(m.weight)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        initer.normal_(m.weight, 1.0, 0.02)
