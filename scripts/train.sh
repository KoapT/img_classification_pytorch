#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

python main.py --arch resnet \
                --depth 18 \
                --save save/road_sign-resnet18-1111 \
                --data road_sign \
                --data_root D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\yolov5_reclass_data_class_view\\guidearrow_genral-2dbox\\ \
                --start-epoch 1 \
                --data_aug \
                --epochs 380 \
                --decay_at_epochs 150 250 330 \
                --batch-size 64 \
                --optimizer adam \
                --lr 0.001 \
                --decay_rate 0.1 \
                --no_nesterov \
                --patience 0 \
                --seed 40 \
                --workers 8 \
                --pretrained \
#                --resume save/road_sign-resnet18-1110/checkpoint.pth.tar \
#                --load-optimizer \
#                --pretrained \
#


#                --eval test \
#                --no-valid \
#                --resume xxxx.pth \
#                --load-optimizer \
#                --fixed-module-flags [] \
#                --cutout \

