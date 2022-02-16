import random
from PIL import Image


class SyncRandomHorizontalFlip:

    def __call__(self, img, seg):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return img, seg
