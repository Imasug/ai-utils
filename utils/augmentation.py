import random
from PIL import Image


class BiCompose:

    def __init__(self, bi_functions):
        self.bi_functions = bi_functions

    def __call__(self, x1, x2):
        for f in self.bi_functions:
            x1, x2 = f(x1, x2)
        return x1, x2


class SyncRandomHorizontalFlip:

    def __call__(self, img, seg):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return img, seg
