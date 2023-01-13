#!/bin/sh
python train.py --net DenseNet40 --lr 0.01 --device cuda --epoch 300 --s
python train.py --net DenseNet100 --lr 0.01 --device cuda --epoch 300 --s
python train.py --net DenseNetBC100 --lr 0.1 --device cuda --epoch 300 --s
