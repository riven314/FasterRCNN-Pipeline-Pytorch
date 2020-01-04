import os
import sys
in_path = os.path.join(os.getcwd(), '..')
train_util_path = os.path.join(os.getcwd(), '..', 'train_utils')
sys.path.append(in_path)
sys.path.append(train_util_path)
print('module path appended!')