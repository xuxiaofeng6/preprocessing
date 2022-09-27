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

Raw_dir = r'G:\Hospital\norm100\cut\ct'
gt_dir = r'G:\Hospital\norm100\cut\gt'
new_dir = r'G:\Hospital\norm100\cut\ct_liver'
pathExist(new_dir)

start = time()
for index,file in enumerate(os.listdir(gt_dir)):
    print(index,file)
    if index >= 0:
        ct = sitk.ReadImage(os.path.join(Raw_dir, file.replace('.nii','_0000.nii')), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        gt = sitk.ReadImage(os.path.join(gt_dir, file), sitk.sitkUInt8)
        gt_array = sitk.GetArrayFromImage(gt)

        ct_array[ct_array <= -100] = -100
        ct_array[ct_array >= 300] = 300

        #ct_array = normalization(ct_array)

        #gt_array[gt_array != 1] = 0

        ct_array[gt_array != 1] = 0

        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(new_dir, file))

