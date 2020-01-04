"""

QUESTIONS
1. when doing img_t.permute(), should be img_t.permute(1, 2, 0) or img_t.permute(2, 1, 0)?
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import init_path


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
    

def write_n_results():
    """
    get n image results and write them in disk
    """
    pass


def img_tensor2np(img_t):
    """
    img_t: one tensor of image 

    REFERENCE:
    - https://discuss.pytorch.org/t/convert-image-tensor-to-numpy-image-array/22887/2
    """
    img = img_t.permute(1, 2, 0).cpu().numpy()
    img = np.asarray(img * 255, dtype = np.uint8)
    return img


def bboxes_tensor2np(boxes_t, img_h):
    """
    boxes_t: list of bboxes tensor

    [[xmin, ymin, xmax, ymax]]
    """
    # prevent overfloat / underfloat
    boxes = np.clip(boxes_t.cpu().numpy(), a_min = 0, a_max = img_h)
    boxes = np.asarray(boxes, dtype = np.uint8)
    return boxes
