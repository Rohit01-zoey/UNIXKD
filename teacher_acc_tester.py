import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import torchvision.transforms as transforms

from dataset import CIFAR100
from models import model_dict

def _accuracy(output, target):
    #returns the correct number of predictions
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
    return correct

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])

trainset = CIFAR100('./data', train=True, transform=transform_train)
valset = CIFAR100('./data', train=False, transform=transform_test)
num_classes = 100

train_loader = DataLoader(trainset, batch_size=64, \
            shuffle=True, num_workers=3, pin_memory=True)
val_loader = DataLoader(valset, batch_size=64, \
            shuffle=False, num_workers=3, pin_memory=True)

teacher_list = ['MobileNetV2', 'resnet8x4', 'resnet20', 'resnet32x4', 
                'ResNet50', 'resnet56', 'ShuffleV1', 'ShuffleV2', 
                'vgg8', 'vgg13', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2']

for teacher_arch in teacher_list:
    teacher_path = f'experiments/teacher_{teacher_arch}'
    ckpt_path = osp.join('{}/ckpt/{}.pth'.format( \
                    teacher_path, 'best'))
    t_model = model_dict[teacher_arch](num_classes=num_classes).cuda()
    state_dict = torch.load(ckpt_path)['state_dict']
    t_model.load_state_dict(state_dict)
    t_model.eval()
    
    #get t_model train and val accuracy
    t_model_acc_train = 0
    train_points = 0
    t_model_acc_val = 0
    val_points = 0
    t_model.eval()
    for _, (x, target, k) in enumerate(train_loader):
        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = t_model(x)
            correct = _accuracy(output, target)
            t_model_acc_train += correct
            train_points += x.size(0)
    t_model_acc_train /= train_points
    
    for x, target in val_loader:
        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = t_model(x)
            correct = _accuracy(output, target)
            t_model_acc_val += correct
            val_points += x.size(0)
    t_model_acc_val /= val_points
    print(f'{teacher_arch} train acc: {t_model_acc_train*100}, val acc: {t_model_acc_val*100}')
    