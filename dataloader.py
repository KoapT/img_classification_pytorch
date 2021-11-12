import torch
import os
import random
import utils
import torchvision.datasets as dset
from torchvision import transforms as T
from torch.utils.data import DataLoader,TensorDataset


class ImageFolder(dset.ImageFolder):
    '''
    借助TensorDataset类，将图片数据直接以tensor方式读入Dataset类中保存，常驻内存，加速后期的load
    '''
    def __init__(self,root, transform=None,target_transform = None,):
        super(ImageFolder, self).__init__(root,transform=transform,
                                          target_transform=target_transform)
        img_tensor_list = []
        target_list = []
        for path, target in self.samples:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            img_tensor_list.append(sample)
            target_list.append(target)
        self.data = TensorDataset(torch.stack(img_tensor_list,0),torch.tensor(target_list))
        del img_tensor_list
        del target_list


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
                   num_workers=6, seed=40, **kwargs):
    train_loader, val_loader, test_loader = None, None, None
    random.seed(seed)

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
            train_set = ImageFolder(os.path.join(data_root, 'train_set'), transform=train_transform).data
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                           random_pick(len(train_set), r=0.1)[1]),
                                       num_workers=num_workers, pin_memory=True, drop_last=True,
                                       prefetch_factor=8)
        if 'val' in splits:
            train_set = ImageFolder(os.path.join(data_root, 'train_set'), transform=test_transform).data
            val_loader = DataLoader(train_set, batch_size=batch_size,
                                     sampler=random_pick(len(train_set), r=0.1)[0],
                                     num_workers=num_workers, pin_memory=True, drop_last=False,
                                     prefetch_factor=8)
        if 'test' in splits:
            test_set = dset.ImageFolder(os.path.join(data_root, 'testset_from_my_device'), transform=test_transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # train_set = dset.ImageFolder('D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\'
    #                              'guidearrow_genral-2dbox\\train_set\\')
    # d = dict([(id, name) for name, id in train_set.class_to_idx.items()])
    # print(d)
    import time

    random.seed(30)
    normalize = T.Normalize(mean=utils.imgs_mean,
                            std=utils.imgs_std)
    test_transform = T.Compose([T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),
                                 T.RandomRotation(degrees=180, expand=True),
                                 T.Resize((64, 64)),
                                 # T.RandomAffine(),
                                 T.ToTensor(),
                                 T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.2, 5), value='random'),
                                 normalize])
    data_root = 'D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\' \
                'guidearrow_genral-2dbox\\'
    t0 = time.time()
    train_set = ImageFolder(os.path.join(data_root, 'train_set'), transform=test_transform)
    train_set = train_set.data
    print(train_set[1])
    print('imgfolder time :', time.time() - t0)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False,
                              # sampler=random_pick(len(train_set), r=0.01)[0],
                              num_workers=1,prefetch_factor=1,
                              pin_memory=True, drop_last=False)
    # for x, y in train_loader:
    #     print(y)
    print('loader time :', time.time() - t0)
    print(train_set)
    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        print(time.time() - end)
        end = time.time()
    print('all time:',time.time()-t0)

    # # calculate mean&std
    # def get_mean_std_value(loader):
    #     channels_sum, channel_squared_sum, num_batches = 0, 0, 0
    #
    #     for data, target in loader:
    #         channels_sum += torch.mean(data, dim=[0, 2, 3])  # shape [n_samples(batch),channels,height,width]
    #         channel_squared_sum += torch.mean(data ** 2,
    #                                           dim=[0, 2, 3])  # shape [n_samples(batch),channels,height,width]
    #         num_batches += 1
    #
    #     # This lo calculate the summarized value of mean we need to divided it by num_batches
    #
    #     mean = channels_sum / num_batches
    #     # 这里将标准差 的公式变形了一下，让代码更方便写一点
    #     std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    #     return mean, std
    #
    #
    # mean, std = get_mean_std_value(train_loader)
    # print('mean = {},std = {}'.format(mean, std))
