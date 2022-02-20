import random
from PIL import Image
import numpy as np


class BiTransforms:

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


class SyncRandomScaledCrop:

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, seg):
        scale = np.random.uniform(*self.scale)
        size = np.array(img.size)
        scaled_size = (size * scale).astype(int)
        img = img.resize(scaled_size, Image.BICUBIC)
        seg = seg.resize(scaled_size)
        if scale > 1.0:
            box1 = np.random.randint(0, scaled_size - size).astype(int)
            box2 = box1 + size
            img = img.crop((*box1, *box2))
            seg = seg.crop((*box1, *box2))
        else:
            box1 = np.random.randint(0, size - scaled_size).astype(int)
            original_img = img.copy()
            img = Image.new(img.mode, tuple(size), 0)
            img.paste(original_img, tuple(box1))
            original_seg = seg.copy()
            seg = Image.new(seg.mode, tuple(size), 0)
            seg.paste(original_seg, tuple(box1))
        return img, seg


class BiTransform:

    def __init__(self, img, seg):
        self.img_transform = img
        self.seg_transform = seg

    def __call__(self, img, seg):
        return self.img_transform(img), self.seg_transform(seg)


class No:

    def __call__(self, x):
        return x
