#!/bin/sh
python train.py --net ResNet32 --device cuda -s -r --epoch 50
python train.py --net ResNet44 --device cuda -s -r --epoch 50
python train.py --net ResNet56 --device cuda -s -r --epoch 50

