"""
sample check models with different min_size set in torchvision.models.detection.fasterrcnn_resnet50_fpn
two model should be the samed, they only different by Transform() method
"""
import os
import unittest

import init_path
from config import cfg
from faster_rcnn import init_pretrain_faster_rcnn
from easydict import EasyDict as edict

# easydict.copy() -> dict
cfg_320 = edict(cfg.copy())
cfg_320.IMG_SIZE = [320, 320] 
assert cfg_320.IMG_SIZE == [320, 320], 'NOT INTENDED'
model_320 = init_pretrain_faster_rcnn(cfg_320)

cfg_600 = edict(cfg.copy())
cfg_600.IMG_SIZE = [600, 600]
assert cfg_600.IMG_SIZE == [600, 600], 'NOT INTENDED'
model_600 = init_pretrain_faster_rcnn(cfg_600)

print('model_320.transform.min_size: {}'.format(model_320.transform.min_size))
print('model_320.transform.image_mean: {}'.format(model_320.transform.image_mean))
print('model_320.transform.image_std: {}'.format(model_320.transform.image_std))
print('model_600.transform.min_size: {}'.format(model_600.transform.min_size))
print('model_600.transform.image_mean: {}'.format(model_600.transform.image_mean))
print('model_600.transform.image_mean: {}'.format(model_320.transform.image_std))