#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

python main.py --arch resnet \
                --depth 18 \
                --save save/road_sign-resnet18-1103 \
                --data road_sign \
                --data_root D:\\iedownload\\Dataset\\yolov5_reclass_data_class_view\\yolov5_reclass_data_class_view\\guidearrow_genral-2dbox\\ \
                --batch-size 64 \
                --eval test \
                --workers 8 \
                --resume save/road_sign-resnet18-1103/checkpoint.pth.tar \

