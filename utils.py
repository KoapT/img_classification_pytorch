import sys
import time
import os
import shutil
import ruamel_yaml
import torch
import torch.nn.functional as F
from importlib import import_module

import numpy as np
from colorama import Fore
import matplotlib.pyplot as plt

label_map = {0: 'F', 1: 'FoL', 2: 'FoR', 3: 'L', 4: 'R', 5: 'other_guides', 6: 'uncertain'}
# imgs_mean= [0.4456, 0.4473, 0.4265]
# imgs_std=[0.2057, 0.2050, 0.2148]
imgs_mean = [0.485, 0.456, 0.406]   # imagenet‘s mean&std
imgs_std=[0.229, 0.224, 0.225]


def getModel(arch, **kargs):
    m = import_module('models.' + arch)
    model = m.createModel(**kargs)
    return model

def create_save_folder(save_path, force=False, ignore_patterns=[]):
    # if os.path.exists(save_path):
    #     print(Fore.RED + save_path + Fore.RESET
    #           + ' already exists!', file=sys.stderr)
    #     if not force:
    #         ans = input('Do you want to overwrite it? [y/N]:')
    #         if ans not in ('y', 'Y', 'yes', 'Yes'):
    #             os.exit(1)
    #     from getpass import getuser
    #     # tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
    #     #                                               os.path.basename(save_path),
    #     #                                               time.time())
    #     # print('move existing {} to {}'.format(save_path, Fore.RED
    #     #                                       + tmp_path + Fore.RESET))
    #     # shutil.copytree(save_path, tmp_path)
    #     shutil.rmtree(save_path)
    try:
        os.makedirs(save_path)
        print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)
    except FileExistsError:
        print('%s has exist.'%save_path)

    # # copy code to save folder
    # if save_path.find('debug') < 0:
    #     shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
    #                     ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
    #                                                   '*.path.tar', '*.pth',
    #                                                   '*.ipynb', '.*', 'data',
    #                                                   'save', 'save_backup',
    #                                                   save_path,
    #                                                   *ignore_patterns))


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs,decay_at_epochs=None):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init

    try:
        decay_at_epochs = list(map(int, decay_at_epochs))
    except ValueError:
        decay_at_epochs=None

    # print('decay_at_epochs:',decay_at_epochs)
    # print('epochs:',num_epochs)
    if decay_at_epochs:
        n = 0
        decay_at_epochs.sort()
        if epoch>=decay_at_epochs[-1]:
            n = len(decay_at_epochs)
        else:
            for i,step in enumerate(decay_at_epochs):
                if epoch<step:
                    n = i
                    break
        lr*=decay_rate**n
    else:
        if epoch >= num_epochs * 0.75:
            lr *= decay_rate**2
        elif epoch >= num_epochs * 0.5:
            lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(parameters, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters, args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters, args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

def optimizer_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def filter_fixed_params(model, fixed_module_flags= []):
    training_params = []
    if fixed_module_flags:
        for n, p in model.named_parameters():
            p.requires_grad=True
            for flag in fixed_module_flags:
                if flag in n:
                    p.requires_grad=False
                    break
            if p.requires_grad:
                training_params.append(p)
        return training_params
    return model.parameters()

def fix_bn(model, fixed_module_flags= []):
    for n, m in model.named_modules():
        if isinstance(m,torch.nn.BatchNorm2d):
            for flag in fixed_module_flags:
                if flag in n:
                    m.eval()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(100. - correct_k*(100./ batch_size))
    return res

###############################################################
# Copied from  https://github.com/uoguelph-mlrg/Cutout
# ECL v2.0 license https://github.com/uoguelph-mlrg/Cutout/blob/master/LICENSE.md

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
###############################################################

###############################################################
# utils to show images and labels&preds on tensorboard.
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    mean= np.array(imgs_mean)
    std = np.array(imgs_std)
    img = img * std[:,None,None] + mean[:,None,None]     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images.cpu()[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            label_map[preds[idx].item()],
            probs[idx] * 100.0,
            label_map[labels[idx].item()]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    # plt.savefig('./save/xx.png')
    return fig
###############################################################