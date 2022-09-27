# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 0024 21:36
# @Author  : Xiaofeng
# @FileName: preprocessing[Label_to_one.py]
# @Software: Pycharm
# @Usages  :

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

Raw_dir = r'G:\Hospital\case231_revised\gt'
new_dir = r'G:\Hospital\case231_revised\gt_portal'
pathExist(new_dir)

start = time()
for file in tqdm(os.listdir(Raw_dir)):
    seg = sitk.ReadImage(os.path.join(Raw_dir, file), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    seg_array[seg_array != 1] = 0
    #seg_array[seg_array == 2] = 1

    # seg_array[seg_array < 3] = 0
    # seg_array[seg_array > 7] = 0
    # seg_array[seg_array == 4] = 0
    # seg_array[seg_array == 5] = 0
    # seg_array[seg_array == 6] = 0
    #
    # seg_array[seg_array == 3] = 0
    # seg_array[seg_array == 7] = 1

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(seg.GetDirection())
    new_seg.SetOrigin(seg.GetOrigin())
    new_seg.SetSpacing(seg.GetSpacing())

    sitk.WriteImage(new_seg, os.path.join(new_dir, file))

