"""
procedures:
- dir for model weights + tensorboar & config.txt
- train, val dataloader

REFERENCE:
1. [github] quick guide on building PyTorch pipeline: https://github.com/Kaixhin/grokking-pytorch?fbclid=IwAR2Ia7TShtqD0uHHR48wrV0U7oS25aabb56Uym7bTcZp9HXy6S0zKsSxfWo
2. [kaggle] train Faster RCNN in PyTorch: https://www.kaggle.com/abhishek/training-fast-rcnn-using-torchvision
3. [kaggle] train Mask RCNN in PyTorch: https://www.kaggle.com/abhishek/train-your-own-mask-rcnn
4. [colab] transfer learning Mask RCNN: https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=ZEARO4B_ye0s
"""
import init_path

import os
import sys
import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from engine import train_one_epoch, evaluate
from faster_rcnn import init_pretrain_faster_rcnn
from config import cfg, write_config
from voc_dataloader import VOCDetection2007

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. set up argument (on cfg, easydict)
parser = argparse.ArgumentParser(description = 'Fine-tuning Faster-RCNN (Res50 backbone) in VOC data format')
parser.add_argument('--voc_base_dir', type = str, help = 'dir to VOC base dir (one showing VOCdevkit)')
parser.add_argument('--worker_n', type = int, default = 10, help = 'no. worker')
parser.add_argument('--s', type = int, help = 'training session (as training ID)')
parser.add_argument('--bs', type = int, default = 8, help = 'batch size for training')
parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs for training')
parser.add_argument('--epochs_per_decay', type = int, default = 10, help = '# epochs to decay learning rate (by 0.1)')
parser.add_argument('--lr', type = float, default = 0.001, help='learning rate')
parser.add_argument('--save_int', type=int, default = 1, help='interval of training for saving models')
args = parser.parse_args()

cfg.SESSION = args.s
cfg.LEARNING_RATE = args.lr
cfg.BATCH_SIZE = args.bs
cfg.EPOCHS = args.epochs
cfg.EPOCHS_PER_DECAY = args.epochs_per_decay
cfg.SAVE_INT = args.save_int
voc_base_dir = args.voc_base_dir
worker_n = args.worker_n


# 2. setup model
print('setting model...')
model = init_pretrain_faster_rcnn(cfg)
model.to(device)


# 3. set up dataloader
print('setting dataloader for traing, val...')
train_data = VOCDetection2007(root = voc_base_dir, image_set = 'train')
val_data = VOCDetection2007(root = voc_base_dir, image_set = 'val')
train_dataloader = DataLoader(train_data, cfg.BATCH_SIZE, 
                              shuffle = True, num_workers = worker_n, 
                              collate_fn = lambda x: tuple(zip(*x)) )
val_dataloader = DataLoader(val_data, cfg.BATCH_SIZE, 
                            shuffle = True, num_workers = worker_n,
                            collate_fn = lambda x: tuple(zip(*x)))


# 4. set up optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
                params, lr = cfg.LEARNING_RATE, momentum=0.9, weight_decay=0.0005
                )
lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size = cfg.EPOCHS_PER_DECAY, gamma = 0.1
                )


# 5a. set up model save dir, logs dir
assert os.path.isdir('models'), '[ERROR] no models dir'
assert os.path.isdir('logs'), '[ERROR] no logs dir'
write_config(cfg, logs_dir = 'logs')
model_save_dir = os.path.join('models', 'session_{}'.format(cfg.SESSION))
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
model_f = 'res50_s{:02d}_e{:03d}.pth'.format(cfg.SESSION, cfg.EPOCHS)
model_temp_path = os.path.join(model_save_dir, model_f)
print('model save path: {}'.format(model_temp_path))


# 5b. start training
for epoch in range(cfg.EPOCHS):
    # train for one epoch, printing every 10 iterations
    train_metrics = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq = 100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    val_metrics = evaluate(model, val_dataloader, device = device)
    if epoch + 1 % cfg.SAVE_INT == 0:
        model_path = model_temp_path.format(cfg.SESSION, epoch)
        torch.save(model.state_dict(), model_path)
        print('model saved: {model_path}')

print('training complete!')




