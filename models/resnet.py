'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torchvision.models as models


def createModel(depth, num_classes, pretrained=True, data='none',**kwargs):
    if data:
        print('Create ResNet-{:d} for {}, {} classes.'.format(depth, data,num_classes))
    if depth == 18:
        # return ResNet(BasicBlock, [2,2,2,2], num_classes)
        if pretrained:
            print('Loading the pretrained parameters resnet18-f37072fd.pth...')
            model = models.resnet18()
            checkpoint = torch.load('./resnet18-f37072fd.pth')
            model.load_state_dict(checkpoint)
            model.fc = torch.nn.Linear(512, num_classes)
            return model

        return models.resnet18(num_classes=num_classes)

    elif depth == 34:
        return models.resnet34(num_classes=num_classes)
    elif depth == 50:
        return models.resnet50(num_classes=num_classes)
    elif depth == 101:
        return models.resnet101(num_classes=num_classes)
    elif depth == 152:
        return models.resnet152(num_classes=num_classes)

if __name__ == '__main__':
    model = createModel(depth=18,num_classes=2,pretrained=False)
    for n,m in model.named_modules():
        print(n,m)
        print('#'*50)