"""
REFERENCE:
1. quick guide on using tensorboard on Pytorch: https://github.com/lanpa/tensorboardX
"""
import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, session, epoch, tb_writer = None, print_freq = 100):
    """
    train one epoch and evaluate loss on training set
    print loss on screen and bookmark in (optionally) tensorboard 

    input:
        device -- cpu mode or cuda mode
        epoch -- int, start from 0
        tb_writer -- tensorboard writer
        print_freq -- int, # iterations (batches) until printint
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter = "  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size = 1, fmt = '{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)
    total_iter = len(data_loader)

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # load in images and gt
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # evaluate losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # single GPU made no difference, for multi GPU
        loss_dict_reduced = utils.reduce_dict(loss_dict) # loss breakdown
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # update print metrics
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # update tensorboard
        if tb_writer is not None:
            iter_n = (epoch) * total_iter + (i + 1)
            tb_dict = {k: v.item() for k, v in loss_dict_reduced.items()}
            tb_dict['loss'] = loss_value
            tb_writer.add_scalars(
                    'session_{:02d}/loss'.format(session), tb_dict, iter_n
                )
    pass


def val_one_epoch(model, optimizer, data_loader, device, session, epoch, tb_writer = None):
    """
    evaluate one epoch of val 
    
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter = "  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size = 1, fmt = '{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)
    total_iter = len(data_loader)
    print_freq = max(int((total_iter - 1) / 2), 1)

    with torch.no_grad():
        for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # load in images and gt
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # evaluate losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # single GPU made no difference, for multi GPU
            loss_dict_reduced = utils.reduce_dict(loss_dict) # loss breakdown
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # update print metrics
            metric_logger.update(loss = losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr = optimizer.param_groups[0]["lr"])
    
            # update tensorboard
            if tb_writer is not None:
                iter_n = (epoch + 1) * total_iter
                tb_dict = {k: v.item() for k, v in loss_dict_reduced.items()}
                tb_dict['loss'] = loss_value
                tb_writer.add_scalars(
                        'session_{:02d}/val_loss'.format(session), tb_dict, iter_n
                    )
    pass


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        # should be unchanged as above
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time = model_time, evaluator_time = evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)