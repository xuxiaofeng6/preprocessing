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

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

Raw_dir = r'G:\Hospital\unnorm231\ZDH_mix_liver\test\0'
new_dir = r'G:\Hospital\unnorm231\ZDH_mix_liver\test\0_cut'
pathExist(new_dir)

start = time()
for file in tqdm(os.listdir(Raw_dir)):
    ct = sitk.ReadImage(os.path.join(Raw_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    ct_array[ct_array <= -100] = -100
    ct_array[ct_array >= 300] = 300

    #ct_array = normalization(ct_array)

    #seg_array[seg_array == 2] = 1

    new_ct = sitk.GetImageFromArray(ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(new_ct, os.path.join(new_dir, file))

