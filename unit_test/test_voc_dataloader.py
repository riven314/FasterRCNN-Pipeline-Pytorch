import os
import sys

import init_path
from voc_dataloader import VOCDetection2007

root = os.path.join(os.getcwd(), '..', '..', 'simulated_data', 'voc_format', 'test')
datasets = VOCDetection2007(root = root, image_set = 'train')

print(datasets[5])