"""
evaluate model performance on validation / test set
- evaluate loss
- write image + bounding box result to a dir

NOTES:
1. in model(images, targets) wrongly alters target['bbox'] value!
2. cv2.rectangle(img, (ymin, xmin), (ymin + w, xmin + h), (255, 0, 0), 2)
"""
import init_path

import os
import sys
import argparse

import cv2
import numpy as np
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
from vis_utils import img_tensor2np, boxes_tensor2np, plot_bbox_on_an_img


voc_base_dir = os.path.join('..', 'simulated_data', 'voc_format', 'mini_easy')
model_path = os.path.join('models', 'session_09', 'res50_s09_e030.pth')
assert os.path.isdir(voc_base_dir), '[ERROR] no such VOC base dir {}'.format(voc_base_dir)
assert os.path.isfile(model_path), '[ERROR] no such model weight {}'.format(model_path)

train_data = VOCDetection2007(root = voc_base_dir, image_set = 'train')
train_dataloader = DataLoader(train_data, 1, shuffle = True, num_workers = 0, collate_fn = lambda x: tuple(zip(*x)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_pretrain_faster_rcnn(cfg)
model.load_state_dict(torch.load(model_path))

model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

data, target = next(iter(train_dataloader))
data = list(d.to(device) for d in data)
target = [{k: v.to(device) for k, v in t.items()} for t in target]
preds = model(data)

# process bbox
img = img_tensor2np(data[0])
boxes = boxes_tensor2np(preds[0]['boxes'], 320)
img = plot_bbox_on_an_img(img, boxes)

plt.imshow(img)
plt.savefig('result.png')

