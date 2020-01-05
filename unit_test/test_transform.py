import os
import init_path

import cv2
import numpy as np 
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from config import cfg
from faster_rcnn import init_pretrain_faster_rcnn
from voc_dataloader import VOCDetection2007, get_transform
import utils
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

from transforms import *
from vis_utils import img_tensor2np, boxes_tensor2np

def plot_img_target(img_t, target_t, w_file):
    boxes = boxes_tensor2np(target_t['boxes'], 320)
    img = img_tensor2np(img_t)
    # plot image with bbox
    new_img = img.copy()
    for box in boxes:
        box = map(int, box)
        xmin, ymin, xmax, ymax = box
        h, w = xmax - xmin, ymax - ymin
        cv2.rectangle(new_img, (ymin, xmin), (ymin + w, xmin + h), (0, 255, 0), 2)
    cv2.imwrite(w_file, new_img)


voc_base_dir = os.path.join('..', '..', 'simulated_data', 'voc_format', 'easy')

# setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#cfg.IMG_SIZE = [320, 320]

model = init_pretrain_faster_rcnn(cfg)
model.to(device)
model.train()

# setup dataloader
dataset = VOCDetection2007(root = voc_base_dir, 
                           image_set = 'val', 
                           transforms = get_transform(True))

# model.train() --> model(imgs, targets) --> loss breakdown
image, target = next(iter(dataset))
print('target: {}'.format(target))
plot_img_target(image, target, 'img_orig.png')

# do transform
hori_trans = RandomHorizontalFlip(1)
vert_trans = RandomVerticalFlip(1)

vert_image, vert_target = vert_trans(image.clone(), edict(target.copy()))
print('vert_target: {}'.format(vert_target))
plot_img_target(vert_image, vert_target, 'img_vert.png')

image, target = next(iter(dataset))
hori_image, hori_target = hori_trans(image.clone(), edict(target.copy()))
print('hori_target: {}'.format(hori_target))
plot_img_target(hori_image, hori_target, 'img_hori.png')






