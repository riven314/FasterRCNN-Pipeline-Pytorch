"""
test writing config log
"""
import os
import unittest

import init_path
from config import cfg, write_config


class MyTestCase(unittest.TestCase):
    """
    set up logs folder inside unit_test
    """
    def setUp(self):
        self.logs_dir = os.path.join('logs')
        self.cfg = cfg
        print('cfg: \n{}'.format(cfg))

    def test1(self):
        with self.assertRaises(AssertionError):
            write_config(self.cfg, self.logs_dir)
    
    def test2(self):
        self.cfg.SESSION = 1
        write_config(self.cfg, self.logs_dir)
    

if __name__ == '__main__':
    unittest.main()

