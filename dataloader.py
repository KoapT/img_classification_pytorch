import torch
import os
import random
import utils
import torchvision.datasets as dset
from torchvision import transforms as T
from torch.utils.data import DataLoader

def random_pick(n, r=0.1):
    '''
    to randomly find r*n iters in the n iterations
    :param n: number of all the iterations
    :param r: ratio to find out
    :return: 2 lists
    '''
    l = list(range(n))
    random.shuffle(l)
    i = int(n * r)
    return l[:i], l[i:]


def getDataloaders(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, use_validset=True, data_root='data', batch_size=64, normalized=True,
                   data_aug=False, cutout=False, n_holes=1, length=16,
                   num_workers=3, seed=40,**kwargs):
    train_loader, val_loader, test_loader = None, None, None
    random.seed(seed)

    if data.find('cifar10') >= 0:
        print('loading ' + data)
        print(config_of_data)
        if data.find('cifar100') >= 0:
            d_func = dset.CIFAR100
            normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                    std=[0.2675, 0.2565, 0.2761])
        else:
            d_func = dset.CIFAR10
            normalize = T.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                    std=[0.2471, 0.2435, 0.2616])
        if data_aug:
            print('with data augmentation')
            aug_trans = [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
            ]
        else:
            aug_trans = []
        common_trans = [T.ToTensor()]
        if normalized:
            print('dataset is normalized')
            common_trans.append(normalize)
        train_compose = aug_trans + common_trans
        if cutout:
            train_compose.append(utils.Cutout(n_holes=n_holes, length=length))
        train_compose = T.Compose(train_compose)
        test_compose = T.Compose(common_trans)

        if use_validset:
            # uses last 5000 images of the original training split as the
            # validation set
            if 'train' in splits:
                train_set = d_func(data_root, train=True, transform=train_compose,
                                   download=True)
                train_loader = DataLoader(
                    train_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        range(45000)),
                    num_workers=num_workers, pin_memory=False)
            if 'val' in splits:
                val_set = d_func(data_root, train=True, transform=test_compose)
                val_loader = DataLoader(
                    val_set, batch_size=batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        range(45000, 50000)),
                    num_workers=num_workers, pin_memory=False)

            if 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = DataLoader(
                    test_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=False)
        else:
            if 'train' in splits:
                train_set = d_func(data_root, train=True, transform=train_compose,
                                   download=True)
                train_loader = DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=False)
            if 'val' in splits or 'test' in splits:
                test_set = d_func(data_root, train=False, transform=test_compose)
                test_loader = DataLoader(
                    test_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=False)
                val_loader = test_loader

    if data.find('road_sign') >= 0:
        print('loading ' + data)
        print(config_of_data)

        normalize = T.Normalize(mean=utils.imgs_mean,
                                std=utils.imgs_std)
        train_transform = T.Compose([T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),
                                     T.RandomRotation(degrees=180, expand=True),
                                     T.Resize((64, 64)),
                                     # T.RandomAffine(),
                                     T.ToTensor(),
                                     T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.2, 5), value='random'),
                                     normalize])
        test_transform = T.Compose([T.Resize((64, 64)),
                                    T.ToTensor(),
                                    normalize])

        def target_transform():
            pass

        if 'train' in splits:
            train_set = dset.ImageFolder(os.path.join(data_root, 'train_set'), transform=train_transform)
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                          random_pick(len(train_set), r=0.1)[1]),
                                      num_workers=num_workers, pin_memory=True, drop_last=False,
                                      prefetch_factor=5)
        if 'val' in splits:
            train_set = dset.ImageFolder(os.path.join(data_root, 'train_set'), transform=test_transform)
            val_loader = DataLoader(train_set, batch_size=batch_size,
                                    sampler=random_pick(len(train_set), r=0.1)[0],
                                    num_workers=num_workers, pin_memory=True, drop_last=False,
                                    prefetch_factor=5)
        if 'test' in splits:
            test_set = dset.ImageFolder(os.path.join(data_root, 'test_set'), transform=test_transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # train_set = dset.ImageFolder('D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\'
    #                              'yolov5_reclass_data_class_view\\guidearrow_genral-2dbox\\train_set\\')
    # d = dict([(id, name) for name, id in train_set.class_to_idx.items()])
    # print(d)
    random.seed(30)
    test_transform = T.Compose([T.Resize((64, 64)),
                                T.ToTensor()])
    data_root = 'D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\' \
                'yolov5_reclass_data_class_view\\guidearrow_genral-2dbox\\'
    train_set = dset.ImageFolder(os.path.join(data_root, 'train_set'), transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False, # sampler=random_pick(len(train_set), r=0.01)[0],
                              pin_memory=False, drop_last=False)
    # for x, y in train_loader:
    #     print(y)


    # calculate mean&std
    def get_mean_std_value(loader):
        channels_sum, channel_squared_sum, num_batches = 0, 0, 0

        for data, target in loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])  # shape [n_samples(batch),channels,height,width]
            # 并不需要求channel的均值
            channel_squared_sum += torch.mean(data ** 2,
                                              dim=[0, 2, 3])  # shape [n_samples(batch),channels,height,width]
            num_batches += 1

        # This lo calculate the summarized value of mean we need to divided it by num_batches

        mean = channels_sum / num_batches
        # 这里将标准差 的公式变形了一下，让代码更方便写一点
        std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean, std


    mean, std = get_mean_std_value(train_loader)
    print('mean = {},std = {}'.format(mean, std))