import os
import sys
path = os.path.join('vision', 'references', 'detection')
sys.path.append(path)
print('module appended: {}'.format(path))