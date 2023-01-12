#!/bin/sh
python train.py --net googlenet --device cuda -s --epoch 100
python train.py --net ResNet20 --device cuda -s  --epoch 100
python train.py --net ResNet32 --device cuda -s  --epoch 100
python train.py --net ResNet44 --device cuda -s  --epoch 100
python train.py --net ResNet56 --device cuda -s  --epoch 100

