import os, shutil
import torch
from importlib import import_module
from torchvision import transforms as T
from PIL import Image
import utils
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
label_map = utils.label_map
mean = utils.imgs_mean
std = utils.imgs_std
mean = [0.485, 0.456, 0.406]   # imagenet‘s mean&std
std=[0.229, 0.224, 0.225]
model = utils.getModel('resnet', num_classes=1000, depth=18, pretrained=True)
# checkpoint = torch.load('./save/road_sign-resnet18-1103/model_best.pth.tar')
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)
target_layer = model.layer4
cam = GradCAMpp(model, target_layer)
input_h, input_w = 224, 224
# img_dir = 'D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\' \
#           'yolov5_reclass_data_class_view\\guidearrow_genral-2dbox\\test_set\\'
img_dir = '../gradcam_plus_plus-pytorch/images'
save_dir = './gradcam_results'

normalize = T.Normalize(mean=mean,
                        std=std)

t = T.Compose([T.Resize((input_h, input_w)),
               T.ToTensor()])


def get_all_imgs(img_dir):
    imgs_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext.lower() in ['.jpg', '.png', '.jpeg']:
                imgpath = os.path.join(root, file)
                imgs_list.append(imgpath)
    return imgs_list


imgs = get_all_imgs(img_dir)
for img in imgs:
    print(img)
    imgname = os.path.split(img)[1]
    im = Image.open(img)
    im = t(im).view(1, 3, input_h, input_w).to(device)
    mask, _ = cam(normalize(im),class_idx=None)
    heatmap, result = visualize_cam(mask, im)
    result = T.ToPILImage()(result)
    result.save(os.path.join(save_dir, imgname))
