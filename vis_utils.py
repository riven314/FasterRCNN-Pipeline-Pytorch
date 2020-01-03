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
        # (x, y) coord is interchanged, also note w, h order
        rect = patches.Rectangle((ymin, xmin), w, h,
                                  linewidth = 1, 
                                  edgecolor = 'r', 
                                  facecolor = 'none')
        ax.add_patch(rect)
    return ax
    
