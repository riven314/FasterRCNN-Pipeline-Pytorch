"""
configuration for data, model and training
"""
import os
from easydict import EasyDict as edict

cfg = edict()

##### Faster RCNN setting ######
# include background
cfg.CLASS_N = 2 
# anchor scales and ratios (list)
cfg.ANCHOR_SCALES = [16, 32, 64, 128, 256]
cfg.ANCHOR_RATIOS = [0.5, 1.0, 2.0]
# number of FPN layers
cfg.FEATURE_N = 5 

##### training setting ######
cfg.LEARNING_RATE = 0.001
cfg.BATCH_SIZE = 12
cfg.EPOCHS = 10
cfg.EPOCHS_PER_DECAY = 10
# interval of epochs for saving models
cfg.SAVE_INT = 1


def write_config(cfg, logs_dir):
    """
    write config log txt file in tensorboard dir
    
    input:
        cfg -- edict, configuration object
        logs_dir -- dir to 'logs' folder 
    """
    assert 'SESSION' in cfg.keys(), '[ERROR] no cfg.SESSION'
    assert os.path.isdir(logs_dir), '[ERROR] logs dir not exist'
    w_dir = os.path.join(logs_dir, 'session_{}'.format(cfg.SESSION))
    if not os.path.isdir(w_dir):
        os.mkdir(w_dir)
    w_path = os.path.join(w_dir, 'config.txt')
    # write config txt file
    f = open(w_path, 'w')
    f.write('CONFIGURATION \n')
    for k, v in cfg.items():
        s = '{}: {} \n'.format(k, v)
        f.write(s)
    f.close()

