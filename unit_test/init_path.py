import os
import sys
m_path = os.path.join(os.getcwd(), '..')
t_path = os.path.join(os.getcwd(), '..', 'train_utils')
sys.path.append(m_path)
sys.path.append(t_path)
print('module path appended')