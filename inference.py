import os, shutil
import torch
from importlib import import_module
from torchvision import transforms as T
from PIL import Image
import utils

label_map = utils.label_map
mean = utils.imgs_mean
std = utils.imgs_std

normalize = T.Normalize(mean=mean,
                        std=std)

t = T.Compose([T.Resize((64, 64)),
               T.ToTensor(),
               normalize])

model = utils.getModel('resnet', num_classes=7, depth=18, pretrained=False)
checkpoint = torch.load('./save/road_sign-resnet18-1103/model_best.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.cuda()


img_dir = 'D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\' \
          'yolov5_reclass_data_class_view\\guidearrow_genral-2dbox\\test_set\\FoR'
for sub in label_map.values():
    if not os.path.isdir(os.path.join(img_dir,sub)):
        os.makedirs(os.path.join(img_dir,sub))
with torch.no_grad():
    for img in os.listdir(img_dir):
        if os.path.isfile(os.path.join(img_dir, img)):
            # print(img)
            im = Image.open(os.path.join(img_dir, img))
            im = t(im).view(1, 3, 64, 64).cuda()
            y = model(im)
            result = torch.argmax(y[0]).item()
            print(result)
            # shutil.copy(os.path.join(img_dir, img),os.path.join(img_dir,label_map[result]))
