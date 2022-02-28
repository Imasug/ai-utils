from distutils.dir_util import copy_tree
from pathlib import Path
from typing import List

from torch.utils.tensorboard import SummaryWriter

from utils.transforms import Transform

import matplotlib.pyplot as plt


class Listener:

    def start(self, target):
        pass

    def pre_epoch(self, epoch, target):
        pass

    def post_epoch(self, epoch, data, target):
        pass

    def end(self):
        pass


class Listeners(Listener):

    def __init__(self, listeners: List[Listener]):
        self.listeners = listeners

    def start(self, target):
        for listener in self.listeners:
            listener.start(target)

    def pre_epoch(self, epoch, target):
        for listener in self.listeners:
            listener.pre_epoch(epoch, target)

    def post_epoch(self, epoch, data, target):
        for listener in self.listeners:
            listener.post_epoch(epoch, data, target)

    def end(self):
        for listener in self.listeners:
            listener.end()


# TODO イテレーションごとの方がよいか？
class TensorBoardLossReporter(Listener):
    TAG = 'loss'

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.train_writer = None
        self.val_writer = None

    def start(self, target):
        folder = self.log_dir.joinpath(target.name)
        train_dir = folder.joinpath('train')
        val_dir = folder.joinpath('val')
        self.train_writer = SummaryWriter(log_dir=train_dir)
        self.val_writer = SummaryWriter(log_dir=val_dir)

    def post_epoch(self, epoch, data, target):
        self.train_writer.add_scalar(self.TAG, data.train_loss, epoch)
        self.val_writer.add_scalar(self.TAG, data.val_loss, epoch)
        self.train_writer.flush()
        self.val_writer.flush()

    def end(self):
        self.train_writer.close()
        self.val_writer.close()


class TensorBoardModelReporter(Listener):

    def __init__(self, log_dir):
        self.log_dir = log_dir

    def start(self, target):
        folder = self.log_dir.joinpath(target.name)
        model_dir = folder.joinpath('model')
        writer = SummaryWriter(log_dir=model_dir)
        data, _ = target.val_data.dataset[0]
        writer.add_graph(target.model, data.unsqueeze(0).to(target.device))
        writer.close()


class PostEpochCopier(Listener):

    def __init__(self, src: Path, dst: Path):
        self.src = src
        self.dst = dst

    def post_epoch(self, epoch, data, target):
        copy_tree(str(self.src), str(self.dst))


class TensorBoardSegmentationInferenceReporter(Listener):

    def __init__(self, log_dir, dataset, inference):
        self.log_dir = log_dir
        self.dataset = dataset
        self.inference = inference
        self.writer = None

    def start(self, target):
        folder = self.log_dir.joinpath(target.name)
        inference_dir = folder.joinpath('inference')
        self.writer = SummaryWriter(log_dir=inference_dir)

    def post_epoch(self, epoch, data, target):
        x, y = 3, len(self.dataset)
        index = 0
        fig = plt.figure(figsize=(15, 5))
        for img, seg in self.dataset:
            # img
            index += 1
            fig.add_subplot(y, x, index)
            plt.imshow(img)

            # seg
            index += 1
            fig.add_subplot(y, x, index)
            plt.imshow(seg)

            # output
            output = self.inference(target.model, img, target.device)
            index += 1
            fig.add_subplot(y, x, index)
            plt.imshow(output)

        self.writer.add_figure(target.name, fig, epoch)
        self.writer.flush()

    def end(self):
        self.writer.close()
