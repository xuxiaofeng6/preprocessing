# -*- coding: utf-8 -*-
# @Time    : 2022/2/22 0022 13:31
# @Author  : Xiaofeng
# @FileName: torch[resize.py]
# @Software: Pycharm
# @Usages  :

import SimpleITK as sitk
import scipy.ndimage as ndimage
import os
from time import time

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def cut_liver():
    raw_path = r'F:\MD\paper\seg\data\public\raw'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'gt')

    new_path = r'F:\MD\paper\seg\data\public\resample'
    new_ct_path = os.path.join(new_path, 'ct')
    new_seg_dir = os.path.join(new_path, 'gt')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_seg_dir)

    start = time()
    for index,file in enumerate(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm
        slice_thickness = 1
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, 1, 1), order=3)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, 1, 1), order=0)

        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_thickness))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_thickness))

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file))

cut_liver()