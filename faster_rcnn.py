"""
prepare Faster RCNN model
- anchor generator
- backbone
- number of feature maps

NOTES:
1. torchvision/models/detection/faster_rcnn.py 
    - transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    - image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

REFERENCE:
1. change pretrained Faster RCNN config in PyTorch: https://github.com/pytorch/vision/issues/978
2. [medium] exact algorithm for Faster-RCNN: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
3. [colab] how to setup fine-tuning Faster RCNN: https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=ZEARO4B_ye0s
4. [github] fasterrcnn_resnet50_fpn source code: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
"""
import os
import sys

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

from config import cfg


def init_pretrain_faster_rcnn(cfg):
    class_n = cfg.CLASS_N
    h, _ = cfg.IMG_SIZE
    # fasterrcnn_resnet50_fpn: pay attention to default min_size = 800, max_size = 1333
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size = h)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_n)
    return model


def __init_pretrain_faster_rcnn(cfg):
    """
    cfg -- dict / edict, configuration object
    """
    # load in key config from cfg
    class_n = cfg.CLASS_N
    anchor_scales = tuple(cfg.ANCHOR_SCALES)
    anchor_ratios = tuple(cfg.ANCHOR_RATIOS)
    feature_n = cfg.FEATURE_N
    # setup backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    # setup RPN 
    anchor_generator = AnchorGenerator(
                        # size refer to length of one side
                        sizes=tuple([anchor_scales for _ in range(feature_n)]),
                        aspect_ratios=tuple([anchor_ratios for _ in range(feature_n)]))
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    # setup RCNN 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_n)
    return model