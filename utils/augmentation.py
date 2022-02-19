import random
from PIL import Image
from torchvision.transforms import *
import torchvision.transforms.functional as F


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

    def __init__(self, *args, **kwargs):
        self.t = RandomRotation(*args, **kwargs)

    def __call__(self, img, seg):
        angle = self.t.get_params(self.t.degrees)
        img = F.rotate(img, angle, self.t.resample, self.t.expand, self.t.center, self.t.fill)
        seg = F.rotate(seg, angle, self.t.resample, self.t.expand, self.t.center, self.t.fill)
        return img, seg
