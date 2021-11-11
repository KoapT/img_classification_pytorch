from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import torch
import torchvision
from utils import AverageMeter, adjust_learning_rate, error, fix_bn, plot_classes_preds
import time
from tensorboardX import SummaryWriter

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        # if self.args.tensorboard:
        #     self.writer = SummaryWriter(os.path.join(self.args.save, 'log_dir'))

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        # fix the specific parameters
        if self.args.fixed_module_flags:
            fix_bn(self.model, self.args.fixed_module_flags)

        lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs, self.args.decay_at_epochs)  # TODO: add custom
        print('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))

        end = time.time()
        for i, (inputs, targets) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # if self.args.tensorboard:
            #     img_grid = torchvision.utils.make_grid(inputs[:4])
            #     self.writer.add_image('input imgs', img_grid)

            inputs = inputs.to(dev)
            targets = targets.to(dev)

            # compute outputs
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure error and record loss
            err1, err5 = error(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(err1.item(), inputs.size(0))
            top5.update(err5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.args.print_freq > 0 and \
                    (i + 1) % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Err@1 {top1.val:.4f}\t'
                      'Err@5 {top5.val:.4f}'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        print('Epoch: {:3d} Train loss {loss.avg:.4f} '
              'Err@1 {top1.avg:.4f}'
              ' Err@5 {top5.avg:.4f}'
              ' Time {batch_time.sum:.3f}'
              .format(epoch, loss=losses, top1=top1, top5=top5, batch_time=batch_time))
        return losses.avg, top1.avg, top5.avg, lr

    def test(self, val_loader, epoch, write=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(dev)
                targets = targets.to(dev)

                # compute outputs
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # measure error and record loss
                err1, err5 = error(outputs.data, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(err1.item(), inputs.size(0))
                top5.update(err5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if self.args.tensorboard:
                #     self.writer.add_figure('predictions vs. actuals',
                #               plot_classes_preds(self.model, inputs, targets),
                #               global_step=epoch * len(val_loader) + i)

        if write:
            with open(os.path.join(self.args.save, 'test_result.txt'), 'a') as f:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), self.args.save, file=f)
                print('Epoch: {:3d} '
                      'val   loss {loss.avg:.4f} '
                      'Err@1 {top1.avg:.4f}'
                      ' Err@5 {top5.avg:.4f}'
                      .format(epoch, loss=losses, top1=top1, top5=top5),
                      end='\n\n', file=f)
        else:
            print('Epoch: {:3d}'
                  ' val   loss {loss.avg:.4f}'
                  ' Err@1 {top1.avg:.4f}'
                  ' Err@5 {top5.avg:.4f}  '
                  'Time {batch_time.sum:.3f}'
                  .format(epoch, loss=losses, top1=top1, top5=top5, batch_time=batch_time))

        return losses.avg, top1.avg, top5.avg
