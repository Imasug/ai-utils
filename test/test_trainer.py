import unittest

import gluoncv
import mxnet as mx
from gluoncv.loss import MixSoftmaxCrossEntropyLoss
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

from tools.trainer import Trainer


class TestTrainer(unittest.TestCase):

    def test(self):
        model = gluoncv.model_zoo.get_psp(dataset='ade20k', backbone='resnet50', pretrained=False)

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        batch_size = 2

        trainset = gluoncv.data.ADE20KSegmentation(split='train', transform=input_transform)
        train_data = gluon.data.DataLoader(
            trainset, batch_size, shuffle=True, last_batch='rollover',
            num_workers=batch_size)

        valset = gluoncv.data.ADE20KSegmentation(split='val', transform=input_transform)
        val_data = gluon.data.DataLoader(
            valset, batch_size, shuffle=True, last_batch='rollover',
            num_workers=batch_size)

        criterion = MixSoftmaxCrossEntropyLoss(aux=True)

        lr_scheduler = gluoncv.utils.LRScheduler(mode='poly', base_lr=0.001,
                                                 nepochs=50, iters_per_epoch=len(train_data), power=0.9)

        kv = mx.kv.create('local')
        optimizer = gluon.Trainer(model.collect_params(), 'sgd',
                                  {'lr_scheduler': lr_scheduler,
                                   'wd': 0.0001,
                                   'momentum': 0.9,
                                   'multi_precision': True},
                                  kvstore=kv)

        def callback(train_loss, val_loss):
            print(f'train loss: {train_loss:.3f}, val loss: {val_loss:.3f}')

        trainer = Trainer(
            model=model,
            epochs=1,
            train_data=train_data,
            val_data=val_data,
            criterion=criterion,
            optimizer=optimizer,
            callback=callback,
        )

        trainer.start()
