"""

QUESTIONS
1. when doing img_t.permute(), should be img_t.permute(1, 2, 0) or img_t.permute(2, 1, 0)?
"""
import init_path

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_and_bbox(img, bbox_ls):
    f, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox in bbox_ls:
        xmin, ymin, xmax, ymax = bbox
        h, w = xmax - xmin, ymax - ymin
        # (xmin, ymin) for [PIL], (xmin, ymin) for cv2
        rect = patches.Rectangle((xmin, ymin), w, h,
                                  linewidth = 1, 
                                  edgecolor = 'r', 
                                  facecolor = 'none')
        ax.add_patch(rect)
    return ax
    

def img_tensor2np(img_t):
    """
    img_t: one tensor of image 

    REFERENCE:
    - https://discuss.pytorch.org/t/convert-image-tensor-to-numpy-image-array/22887/2
    """
    img = img_t.permute(1, 2, 0).cpu().numpy()
    img = np.asarray(img * 255, dtype = np.uint8)
    return img


def boxes_tensor2np(boxes_t, img_h):
    """
    boxes_t: list of bboxes tensor

    [[xmin, ymin, xmax, ymax]]
    """
    # prevent overfloat / underfloat
    boxes = np.clip(boxes_t.cpu().numpy(), a_min = 0, a_max = img_h)
    #boxes = np.asarray(boxes, dtype = np.int32)
    return boxes


def plot_bbox_on_an_img(img, boxes, target):
    """
    input:
        img -- np uint8 array orig image
        boxes -- array of [xmin, ymin, xmax, ymax]
        target -- list of target (tensor)
    output:
        img -- np uint8 array image with bbox (change in place)
    """
    if boxes is not None:
        for box in boxes:
            box = map(int, box)
            xmin, ymin, xmax, ymax = box
            h, w = xmax - xmin, ymax - ymin
            cv2.rectangle(img, (ymin, xmin), (ymin + w, xmin + h), (0, 255, 0), 1)
            #cv2.putText(image, 'Fedex', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    if target is not None:
        for box in target[0]['boxes'].tolist():
            box = map(int, box)
            xmin, ymin, xmax, ymax = box
            h, w = xmax - xmin, ymax - ymin
            cv2.rectangle(img, (ymin, xmin), (ymin + w, xmin + h), (255, 0, 0), 1)
    return img

    