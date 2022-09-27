# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 0005 12:48
# @Author  : Xiaofeng
# @FileName: torch[read_h5]
# @Software: Pycharm
# @Usages  :

import os
import numpy as np
import h5py

# path = r'J:\Dataset\LITS\h5\train_1.h5'
path = r'J:\Dataset\SegWithDistMap\case231\h5\ABD_001.h5'

h5f = h5py.File(path, 'r')
image = h5f['image'][:]

print(image.shape)
print(np.max(image))
print(np.min(image))