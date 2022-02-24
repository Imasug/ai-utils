from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

TAG = 'loss'


# TODO イテレーションごとの方がよいか？
class TensorBoardLossReporter:

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.train_writer = None
        self.val_writer = None

    def start(self):
        train_dir = self.log_dir.joinpath('train')
        val_dir = self.log_dir.joinpath('val')
        self.train_writer = SummaryWriter(log_dir=train_dir)
        self.val_writer = SummaryWriter(log_dir=val_dir)

    def pre_epoch(self, epoch, target):
        pass

    def post_epoch(self, epoch, data, target):
        self.train_writer.add_scalar(TAG, data.train_loss, epoch)
        self.val_writer.add_scalar(TAG, data.val_loss, epoch)

    def end(self):
        self.train_writer.close()
        self.val_writer.close()
