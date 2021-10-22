# -*- coding: utf-8 -*-
# @Time    : 2021/6/23 0023 16:45
# @Author  : Xiaofeng
# @FileName: preprocessing[labels.py]
# @Software: Pycharm
# @Usages  : 将两种类型血管合到一个mask上 背景==0 门静脉==1 肝静脉==2

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage


raw_dir = '/data/xiaofeng/case50_cut_z'
ct_dir = os.path.join(raw_dir,'ct')
#vein_dir = os.path.join(raw_dir,'portalvein')
vein_dir = os.path.join(raw_dir,'vessel')
vena_dir = os.path.join(raw_dir,'venacava')

#Target_dir = r'G:\Public Dataset\Radiology\3Dircadb\Train Data Cut'
new_label_dir = os.path.join(raw_dir,'vessel_')
os.mkdir(new_label_dir)


start = time()
for file in tqdm(os.listdir(vein_dir)):
    seg1 = sitk.ReadImage(os.path.join(vein_dir, file), sitk.sitkUInt8)
    seg1_array = sitk.GetArrayFromImage(seg1)
    seg_array = np.copy(seg1_array)

    #seg2 = sitk.ReadImage(os.path.join(vena_dir, file), sitk.sitkUInt8)
    #seg2_array = sitk.GetArrayFromImage(seg2)

    #门静脉和肝静脉的label 背景==0 门静脉==1 肝静脉==2
    #seg_array[seg1_array > 0] = 1
    #seg_array[seg2_array > 0] = 2

    seg_array[seg1_array == 1] = 4
    seg_array[seg1_array == 2] = 4
    seg_array[seg1_array == 3] = 1
    seg_array[seg_array == 4] = 2

    # 最终将数据保存为nii

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(seg1.GetDirection())
    new_seg.SetOrigin(seg1.GetOrigin())
    new_seg.SetSpacing(seg1.GetSpacing())

    sitk.WriteImage(new_seg, os.path.join(new_label_dir, file))

