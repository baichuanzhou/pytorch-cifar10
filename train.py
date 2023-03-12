"""
This file encodes the workflow to train nets from models directory.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as dset
import torchvision.transforms as T

import datetime
import time
import os
import sys
import argparse
from utils import *

import matplotlib.pyplot as plt

# Handling the interface we designed
parser = argparse.ArgumentParser(description='Models implemented using Pytorch on cifar10')
parser.add_argument("--lr", type=float, action="store", default=0.01, help="learning rate, default 0.01")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--net", "-n", type=str, required=True, help="choose the neural network from our collections")
parser.add_argument("--device", "-d", type=str, required=True, help="choose the preferred device to work on")
parser.add_argument("--dtype", type=str, default="torch.float32", help="set the dtype for the model")
parser.add_argument("--save", "-s", action="store_true", default=False, help="save the parameters of the model")
parser.add_argument("--optim", action="store", type=str, default="sgd", help="choose the optimization method")
parser.add_argument("--epoch", action="store", type=int, default=1, help="define the epochs you want for training")

# Now we preprocess the cifar-10 data provided by pytorch

NUM_TRAIN = 45000
transform = T.Compose([
    T.ToTensor(),
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]
)

cifar10_train = dset.CIFAR10('./datasets', train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

# We use 5000 samples as validation set.
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True, transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=128, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=128)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == "__main__":
    args = parser.parse_args()

    # set the device for training
    if args.device != 'cpu' and args.device != 'cuda':
        raise ValueError("Device must be cuda or cpu")

    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # set the dtype
    if args.dtype:
        try:
            dtype = eval(args.dtype)
            torch.ones(1, dtype=dtype)
        except:
            raise ValueError("dtype must be allowed by PyTorch")

    # set the model
    acc_his = []
    loss_his = []
    model_name = args.net
    model, model_name = get_model(model_name)

    if str.lower(args.optim) == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif str.lower(args.optim) == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    elif str.lower(args.optim) == "rmsprop":
        optimizer = optim.RMSprop(params=model.parameters(), lr=args.lr)
    else:
        optimizer = eval("optim." + args.optim + "(params=model.parameters, lr=args.lr)")

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45, 70, 90, 100])

    epochs = args.epoch
    if "DenseNet" in model_name:
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs * 0.5, epochs * 0.75])

    if "ViT" in model_name:
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    if args.resume:
        resume_path = os.path.join('checkpoints/%s.pth' % model_name)
        assert os.path.isfile(resume_path), "path for file %s does not exist" % resume_path
        print("--> Resuming from checkpoint for", model_name)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['params'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch']

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45, 70, 90, 100],
                                                      last_epoch=last_epoch - 1)
        if 'accuracy_history' in checkpoint.keys():
            acc_his = checkpoint['accuracy_history']
        if 'loss_history' in checkpoint.keys():
            loss_his = checkpoint['loss_history']

        print("%s has been trained for %d epochs" % (model_name, checkpoint['epoch']))

    print("Start training at:", datetime.datetime.now())
    start_time = time.time()

    train(model,
          optimizer,
          loader_train=loader_train,
          loader_val=loader_val,
          loss_his=loss_his,
          acc_his=acc_his,
          epoch=epochs,
          lr_scheduler=lr_scheduler)

    end_time = time.time()
    print("Training took: ------- %s seconds -------" % (end_time - start_time))

    print("Start testing at:", datetime.datetime.now())
    start_time = time.time()
    acc = check_accuracy(model, loader_test)
    end_time = time.time()
    print("Testing took: ------- %s seconds -------" % (end_time - start_time))

    if args.save:
        save(model_name,
             model=model,
             optimizer=optimizer,
             loss_his=loss_his,
             acc_his=acc_his,
             epochs=epochs)

    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.title('Validation Correctness')
    plt.plot(acc_his, '-o', color='b')

    plt.subplot(2, 1, 2)
    plt.xlabel('Iteration')
    plt.title("Training Loss")
    plt.plot(loss_his, '--', color='b')

    plt.gcf().set_size_inches(15, 15)

    try:
        fig_path = os.path.join('./figs/%s.png' % model_name)
        plt.savefig(fig_path)
    except FileNotFoundError:
        os.mkdir('./figs')
        fig_path = os.path.join('./figs/%s.png' % model_name)
        plt.savefig(fig_path)
