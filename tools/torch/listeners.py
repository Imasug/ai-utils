from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Listener:

    def start(self, target):
        pass

    def pre_epoch(self, epoch, target):
        pass

    def prost_epoch(self, epoch, data, target):
        pass

    def end(self):
        pass


TAG = 'loss'


# TODO イテレーションごとの方がよいか？
class TensorBoardLossReporter(Listener):

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
        self.train_writer.add_scalar(TAG, data.train_loss, epoch)
        self.val_writer.add_scalar(TAG, data.val_loss, epoch)

    def end(self):
        self.train_writer.close()
        self.val_writer.close()


class TorchTensorBoardModelReporter(Listener):

    def __init__(self, log_dir):
        self.log_dir = log_dir

    def start(self, target):
        folder = self.log_dir.joinpath(target.name)
        model_dir = folder.joinpath('model')
        writer = SummaryWriter(log_dir=model_dir)
        data, _ = target.val_data.dataset[0]
        writer.add_graph(target.model, data)
        writer.close()
