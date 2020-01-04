"""
evaluate model performance on validation / test set
- evaluate loss
- write image + bounding box result to a dir

NOTES:
1. in model(images, targets) wrongly alters target['bbox'] value!
"""
import init_path

import os
import sys
import argparse

import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_utils.engine import train_one_epoch, evaluate
from config import cfg
from faster_rcnn import init_pretrain_faster_rcnn
from voc_dataloader import VOCDetection2007

voc_base_dir = os.path.join('..', 'simulated_data', 'voc_format', 'easy')
model_path = os.path.join('models', 'session_1', 'res50_s01_e001.pth')
assert os.path.isdir(voc_base_dir), '[ERROR] no such VOC base dir {}'.format(voc_base_dir)
assert os.path.isfile(model_path), '[ERROR] no such model weight {}'.format(model_path)

val_data = VOCDetection2007(root = voc_base_dir, image_set = 'val')
val_dataloader = DataLoader(val_data, 1, shuffle = True, num_workers = 0, collate_fn = lambda x: tuple(zip(*x)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_pretrain_faster_rcnn(cfg)
model.load_state_dict(torch.load(model_path))

model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

model.eval()

data, target = next(iter(val_dataloader))
data = list(d.to(device) for d in data)
#target = [{k: v.to(device) for k, v in t.items()} for t in target]
#loss_dict = model(data, target)
#losses = sum(loss for loss in loss_dict.values())

images, targets = next(iter(val_dataloader))
images = list(image.to(device) for image in images)
new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#loss_dict = model(images, new_targets)
#losses = sum(loss for loss in loss_dict.values())