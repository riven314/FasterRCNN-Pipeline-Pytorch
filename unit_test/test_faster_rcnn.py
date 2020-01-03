import os
import unittest

import init_path
from config import cfg
from faster_rcnn import init_pretrain_faster_rcnn


model = init_pretrain_faster_rcnn(cfg)
