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

from voc_dataloader import VOCDetection2007, get_transform
from vis_utils import img_tensor2np, boxes_tensor2np, plot_bbox_on_an_img


def inference_n_img(model, device, voc_base_dir, write_dir, n, mode = 'val'):
    """
    load n images and plot their bounding box prediction result
    write into disk and disable data augmentation

    input:
        write_dir -- dir for writing images with bounding box
        n -- # image to be plotted
        mode -- str, 'train'/ 'val' (use train data or val data)
    """
    assert mode in ['train', 'val'], '[ERROR] mode argument is wrong'
    temp_png = 'result_{:04d}.png'
    cpu_device = torch.device("cpu")

    model.to(device)
    model.eval()
    dataset = VOCDetection2007(
        root = voc_base_dir, image_set = mode, transforms = get_transform(False)
        )
    dataloader = DataLoader(
        dataset, 1, shuffle = True, num_workers = 0, collate_fn = lambda x: tuple(zip(*x))
        )
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if i >= n:
                break
            data = list(d.to(device) for d in data)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]
            preds = model(data)
            target = [{k: v.to(cpu_device) for k, v in t.items()} for t in target]
            preds = [{k: v.to(cpu_device) for k, v in t.items()} for t in preds]
            img = img_tensor2np(data[0])
            boxes = preds[0]['boxes']
            boxes = boxes_tensor2np(boxes, img_h = data[0].shape[-1])
            bbox_img = plot_bbox_on_an_img(img, boxes, target)
            w_path = os.path.join(write_dir, temp_png.format(i))
            cv2.imwrite(w_path, bbox_img)

