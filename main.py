#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from colorama import Fore

import config
from dataloader import getDataloaders
from utils import getModel, save_checkpoint, get_optimizer, optimizer_device, create_save_folder, filter_fixed_params
from args import arg_parser, arch_resume_names
from train_val import Trainer
from tensorboardX import SummaryWriter

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    # parse arg and start experiment
    global args
    best_err1 = 100.
    best_epoch = 0
    now_epoch = 0

    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            if args.start_epoch <= 1:
                args.start_epoch = checkpoint['epoch'] + 1
            now_epoch = args.start_epoch - 1
            best_epoch = checkpoint['best_epoch']
            best_err1 = checkpoint['best_err1']
            print('pre best err1:{} @epoch{}'.format(best_err1, best_epoch))
            # for name in arch_resume_names:
            #     if name in vars(args) and name in vars(old_args):
            #         setattr(args, name, getattr(old_args, name))
            model = getModel(**vars(args))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
            model = model.to(dev)
            training_params = filter_fixed_params(model, args.fixed_module_flags)
            optimizer = get_optimizer(training_params, args)

            if args.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_device(optimizer, dev)
                print('optimizer.state_dict():')
                print(optimizer.state_dict()["param_groups"])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))
        model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        model = model.to(dev)
        training_params = filter_fixed_params(model, args.fixed_module_flags)
        optimizer = get_optimizer(training_params, args)

    cudnn.benchmark = True
    # check if the folder exists
    create_save_folder(args.save, args.force)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().to(dev)

    # set random seed
    torch.manual_seed(args.seed)

    trainer = Trainer(model, criterion, optimizer, args)

    # create dataloader
    if args.evaluate == 'train':
        train_loader, _, _ = getDataloaders(
            splits=('train'), **vars(args))
        trainer.test(train_loader, now_epoch)
        return
    elif args.evaluate == 'val':
        _, val_loader, _ = getDataloaders(
            splits=('val'), **vars(args))
        trainer.test(val_loader, now_epoch)
        return
    elif args.evaluate == 'test':
        _, _, test_loader = getDataloaders(
            splits=('test'), **vars(args))
        trainer.test(test_loader, now_epoch, write=True)
        return
    else:
        train_loader, val_loader, _ = getDataloaders(
            splits=('train', 'val'), **vars(args))

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)

    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))
    f_log.flush()
    f_log.close()
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err5']
    if args.tensorboard:
        writer = SummaryWriter(os.path.join(args.save, 'log_dir'))

    for epoch in range(args.start_epoch, args.epochs + 1):

        # simulate 10 fold validation
        if epoch % (args.epochs / 10) == 0:
            args.seed += 5
            train_loader, val_loader, _ = getDataloaders(
                splits=('train', 'val'), **vars(args))

        # train for one epoch
        train_loss, train_err1, train_err5, lr = trainer.train(
            train_loader, epoch)

        # evaluate on validation set
        val_loss, val_err1, val_err5 = trainer.test(val_loader, epoch)

        if args.tensorboard:
            writer.add_scalar('Lr', lr, epoch)
            writer.add_scalars('Err/err1', {'train_err1': train_err1,
                                            'val_err1': val_err1}, epoch)
            writer.add_scalars('Err/err5', {'train_err5': train_err5,
                                            'val_err5': val_err5}, epoch)
            writer.add_scalars('Loss', {'train_loss': train_loss,
                                        'val_loss': val_loss}, epoch)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)

        # remember best err@1 and save checkpoint
        is_best = val_err1 < best_err1
        if is_best:
            best_err1 = val_err1
            best_epoch = epoch
        print(Fore.GREEN + 'Best var_err1 {} @ ep{}'.format(best_err1, best_epoch) +
              Fore.RESET)
        # test_loss, test_err1, test_err1 = validate(
        #     test_loader, model, criterion, epoch, True)
        # save test
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'arch': args.arch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_err1': best_err1}, is_best, args.save)
        if not is_best and epoch - best_epoch >= args.patience > 0:
            break
    with open(os.path.join(args.save, 'scores.tsv'), 'a') as f:
        print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch), file=f)


if __name__ == '__main__':
    main()
