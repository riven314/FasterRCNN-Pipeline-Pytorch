"""
sample an epoch and look at dataloader and loss

OBSERVATION:
- even though model.transform.min_size = 600, preds['boxes'] is still bounded by (320, 320)
"""
import os
import init_path

import torch
import torchvision
from torch.utils.data import DataLoader

from config import cfg
from faster_rcnn import init_pretrain_faster_rcnn
from voc_dataloader import VOCDetection2007
import utils
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator


voc_base_dir = os.path.join('..', '..', 'simulated_data', 'voc_format', 'easy')

# setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.IMG_SIZE = [600, 600]

model = init_pretrain_faster_rcnn(cfg)
model.to(device)
model.train()
metric_logger = utils.MetricLogger(delimiter="  ")
metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
print('model.transform.min_size: {}\n\n'.format(model.transform.min_size))

# setup dataloader
dataset = VOCDetection2007(root = voc_base_dir, image_set = 'train')
dataloader = DataLoader(dataset, 3, 
                        shuffle = True, num_workers = 0, 
                        collate_fn = lambda x: tuple(zip(*x)) )


# model.train() --> model(imgs, targets) --> loss breakdown
images, targets = next(iter(dataloader))
print('images shape: {} \n\n'.format(images[0].shape))

images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
loss_dict = model(images, targets)
loss_dict_reduced = utils.reduce_dict(loss_dict)
losses_reduced = sum(loss for loss in loss_dict_reduced.values())
# gross sum of loss
loss_value = losses_reduced.item()

print('loss_dict.keys: {}'.format(loss_dict.keys()))
print('loss_dict: {}'.format(loss_dict))
print('loss classifier: {}'.format(loss_dict['loss_classifier'].cpu().tolist()))
print('losses_reduced: {}'.format(losses_reduced))
print('loss_value: {}\n\n'.format(loss_value))


# model.eval() --> model(imgs) --> model post-processed bbox predictions
model.eval()
preds = model(images)
print('preds[0].keys(): {}'.format(preds[0].keys()))
print('preds[0][boxes]: {}'.format(preds[0]['boxes'][:6]))
print('preds[0][labels]: {}'.format(preds[0]['labels'][:6]))