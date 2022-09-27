# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 0005 14:20
# @Author  : Xiaofeng
# @FileName: torch[rotate]
# @Software: Pycharm
# @Usages  :

import os
import SimpleITK as sitk
import numpy as np

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

raw_dir = r'J:\Dataset\SegWithDistMap\case231\nii\gt'
out_dir = r'J:\Dataset\SegWithDistMap\case231\nii\gt_rotate1'
pathExist(out_dir)

for index,file in enumerate(os.listdir(raw_dir)):
    ct = sitk.ReadImage(os.path.join(raw_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    print(ct_array.shape)

    new_array = np.transpose(ct_array, (1, 2, 0))
    #new_array = np.transpose(ct_array, (2, 0, 1))

    new_seg = sitk.GetImageFromArray(new_array)

    # new_seg.SetDirection(ct.GetDirection())
    # new_seg.SetOrigin(ct.GetOrigin())
    # new_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(new_seg, os.path.join(out_dir, file))