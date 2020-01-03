"""
loading data in VOC format. reference from (with a few modification):
https://github.com/pytorch/vision/blob/07cbb46aba8569f0fac95667d57421391e6d36e9/torchvision/datasets/voc.py#L213
"""
import os
import sys
import collections

import cv2
import numpy as np
from PIL import Image
from easydict import EasyDict as edict

import xml.etree.cElementTree as ET
import torch
from torch.utils.data import Dataset


class VOCDetection2007(Dataset):
    """
    VOC 2007 format

    input:
        root (string): Root directory of the VOC Dataset. (dir storing VOCdevkit)
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        height: int, size input image, assume square i.e. size (height, height, 3)
        file_ext: default '.tif', image file extension
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, image_set = 'train', height = 320, file_ext = '.tif', transforms = None):
        #super(VOCDetection2007, self).__init__(root, transforms)
        valid_sets = ["train", "trainval", "val"]
        assert image_set in valid_sets, '[ERROR] invalid argument image_set'
        base_dir = os.path.join('VOCdevkit', 'VOC2007')
        voc_root = os.path.join(root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        assert os.path.isdir(voc_root), '[ERROR] Dataset not found or corrupted.'

        splits_dir = os.path.join(voc_root, 'ImageSets', 'Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        # get list of path to images and annotation files
        self.images = [os.path.join(image_dir, x + file_ext) for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations)), '[ERROR] image no. != annotation file no.' 

        self.transforms = transforms
        self.height = height

    def __getitem__(self, index):
        """
        input:
            index (int): Index
        output:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        voc_dict = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        target = self.dict_to_target(voc_dict, index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def dict_to_target(self, voc_dict, index):
        """
        construct target 

        input:
            voc_dict -- dict/ easydict: all nested contents of the target xml file
        output:
            target -- dict/ easydict: [boxes, labels, masks, image_id, area, iscrowd]
        """
        # build boxes
        boxes = []
        for obj_dict in voc_dict.annotation.object:
            xmin = max(int(obj_dict.bndbox.xmin), 0)
            ymin = max(int(obj_dict.bndbox.ymin), 0)
            xmax = max(int(obj_dict.bndbox.xmax), 0)
            ymax = max(int(obj_dict.bndbox.ymax), 0)
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        # build labels
        obj_n = len(voc_dict.annotation.object)
        labels = torch.ones((obj_n,), dtype = torch.int64)
        # build image_id
        image_id = torch.tensor([index])
        # build area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # build iscrowd
        iscrowd = torch.zeros((obj_n,), dtype = torch.int64)
        # build target dict
        target = edict()
        target.boxes = boxes
        target.labels = labels
        target.image_id = image_id
        target.area = area
        target.iscrowd = iscrowd
        return target
        
    def parse_voc_xml(self, node):
        """
        parse xml file into a dict (easydict)
        """
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        voc_dict = edict(voc_dict)
        return voc_dict
