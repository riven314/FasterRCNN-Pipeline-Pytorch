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
from tensorboardX import SummaryWriter

from engine import train_one_epoch, val_one_epoch, evaluate
from faster_rcnn import init_pretrain_faster_rcnn
from config import cfg, write_config
from voc_dataloader import VOCDetection2007, get_transform
from eval import inference_n_img


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. set up argument (on cfg, easydict)
parser = argparse.ArgumentParser(description = 'Fine-tuning Faster-RCNN (Res50 backbone) in VOC data format')
parser.add_argument('--voc_base_dir', type = str, help = 'dir to VOC base dir (one showing VOCdevkit)')
parser.add_argument('--worker_n', type = int, default = 10, help = 'no. worker')
parser.add_argument('--s', type = int, help = 'training session (as training ID)')
parser.add_argument('--bs', type = int, default = 8, help = 'batch size for training')
parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs for training')
parser.add_argument('--epochs_per_decay', type = int, default = 10, help = '# epochs to decay learning rate (by 0.1)')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--save_int', type = int, default = 1, help = 'interval of training for saving models')
parser.add_argument('--infer_img_n', type = int, default = 0, help = '# of predicted images to be written')
parser.add_argument('--use_tb', help='whether use tensorboard', action='store_true')
parser.add_argument('--use_aug', help='whether use image augmentation (horizontal / vertical flip)', action='store_true')
args = parser.parse_args()

cfg.SESSION = args.s
cfg.LEARNING_RATE = args.lr
cfg.BATCH_SIZE = args.bs
cfg.EPOCHS = args.epochs
cfg.EPOCHS_PER_DECAY = args.epochs_per_decay
cfg.SAVE_INT = args.save_int
cfg.USE_DATA_AUG = args.use_aug
voc_base_dir = args.voc_base_dir
worker_n = args.worker_n


# 2. setup model
print('setting model...')
model = init_pretrain_faster_rcnn(cfg)
model.to(device)


# 3. set up dataloader
# tensor: (C, H, W)
print('setting dataloader for traing, val...')
if args.use_aug:
    print('data augmentation mode is on')
    train_transform = get_transform(is_aug = True)
else:
    train_transform = get_transform(is_aug = False)

train_data = VOCDetection2007(root = voc_base_dir, image_set = 'train', transforms = train_transform)
val_data = VOCDetection2007(root = voc_base_dir, image_set = 'val', transforms = get_transform(is_aug = False))
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


# 5. set up model save dir, logs dir, tensorboard
assert os.path.isdir('models'), '[ERROR] no models dir'
assert os.path.isdir('logs'), '[ERROR] no logs dir'
write_config(cfg, logs_dir = 'logs')
model_save_dir = os.path.join('models', 'session_{:02d}'.format(cfg.SESSION))
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
model_f = 'res50_s{:02d}_e{}.pth'.format(cfg.SESSION, '{:03d}')
model_temp_path = os.path.join(model_save_dir, model_f)
print('model save path: {}'.format(model_temp_path))

tb_writer = None
if args.use_tb:
    tb_writer = SummaryWriter('logs')


# 6. start training
for epoch in range(cfg.EPOCHS):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_dataloader, device, cfg.SESSION, epoch, tb_writer = tb_writer, print_freq = 100)
    # evaluate val loss (val_one_epoch) and val mAP, mAR (evaluate)
    val_one_epoch(model, optimizer, val_dataloader, device, cfg.SESSION, epoch, tb_writer = tb_writer)
    evaluate(model, val_dataloader, device = device)
    # update the learning rate
    lr_scheduler.step()
    if (epoch + 1) % cfg.SAVE_INT == 0:
        model_path = model_temp_path.format(epoch + 1)
        torch.save(model.state_dict(), model_path)
        print('model saved: {}'.format(model_path))


# 7. plotting results for training set and test set
if args.infer_img_n != 0:
    write_train_img_dir = os.path.join('logs', 'session_{:02d}'.format(cfg.SESSION), 'train_results')
    write_val_img_dir = os.path.join('logs', 'session_{:02d}'.format(cfg.SESSION), 'val_results')
    if not os.path.isdir(write_train_img_dir):
        os.mkdir(write_train_img_dir)
    if not os.path.isdir(write_val_img_dir):
        os.mkdir(write_val_img_dir)
    inference_n_img(model, device, voc_base_dir, write_train_img_dir, n = args.infer_img_n, mode = 'train')
    inference_n_img(model, device, voc_base_dir, write_val_img_dir, n = args.infer_img_n, mode = 'val')


# 8. final save model 
model_path = model_temp_path.format(epoch + 1)
torch.save(model.state_dict(), model_path)
print('model_saved: {}'.format(model_path))
print('training complete!')

if args.use_tb:
    tb_writer.close()

