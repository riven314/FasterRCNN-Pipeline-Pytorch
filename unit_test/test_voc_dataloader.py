"""
sample check some index output from VOCDetection2007
also visualize image with bounding boxes to see if bounding boxes are correct

REFERENCE:
1. drawing bounding boxes on an image: https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import init_path
from voc_dataloader import VOCDetection2007

root = os.path.join(os.getcwd(), '..', '..', 'simulated_data', 'voc_format', 'easy')
datasets = VOCDetection2007(root = root, image_set = 'train')

print(datasets[5])

# (C, H, W)
img = np.asarray(datasets[5][0].permute(1, 2, 0).numpy() * 255, dtype = np.uint8)
bbox_ls = datasets[5][1].boxes.numpy() # tensor to numpy array
area_ls = datasets[5][1].area.numpy()

f,ax = plt.subplots(1)
ax.imshow(img)
for area, bbox in zip(area_ls, bbox_ls):
    xmin, ymin, xmax, ymax = bbox
    print('bbox: ({}, {}), ({}, {})'.format(xmin, ymin, xmax, ymax))
    print('area: {}'.format(area))
    h, w = xmax - xmin, ymax - ymin
    # (x, y) coord is interchanged, also note w, h order
    rect = patches.Rectangle((xmin, ymin), w, h,
                              linewidth = 1, edgecolor = 'r', facecolor = 'none')
    ax.add_patch(rect)

plt.show()
