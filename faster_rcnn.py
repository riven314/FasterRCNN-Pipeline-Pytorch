"""
prepare Faster RCNN model
- anchor generator
- backbone
- number of feature maps

REFERENCE:
1. change pretrained Faster RCNN config in PyTorch: https://github.com/pytorch/vision/issues/978
2. [medium] exact algorithm for Faster-RCNN: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
"""
import os
import sys

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


def init_pretrain_faster_rcnn(class_n, 
                              anchor_sizes = (16, 32, 64, 128, 256), 
                              aspect_ratio = (0.5, 1.0, 2.0),
                              feature_n = 5):
    """
    anchor_sizes -- tuple of anchor size (size = a length of a square)
    aspect_ratio -- 
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    anchor_generator = AnchorGenerator(
                        # size refer to length of one side
                        sizes=tuple([anchor_sizes for _ in range(feature_n)]),
                        aspect_ratios=tuple([aspect_ratio for _ in range(feature_n)]))
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_n)
    return model