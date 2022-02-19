import random
from PIL import Image
from torchvision.transforms import *
import numpy as np


class BiCompose:

    def __init__(self, bi_functions):
        self.bi_functions = bi_functions

    def __call__(self, x1, x2):
        for f in self.bi_functions:
            x1, x2 = f(x1, x2)
        return x1, x2


class SyncRandomHorizontalFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, seg):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return img, seg


class SyncRandomRotation:

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, seg):
        angle = np.random.uniform(*self.degrees)
        img = img.rotate(angle)
        seg = seg.rotate(angle)
        return img, seg


class SyncRandomScale:

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, seg):
        scale = np.random.uniform(*self.scale)
        size = (np.array(img.size) * scale).astype(int)
        img = img.resize(size, Image.BICUBIC)
        seg = seg.resize(size)
        return img, seg
