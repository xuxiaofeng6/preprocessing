# -*- coding: utf-8 -*-
# @Time    : 2022/2/22 0022 13:51
# @Author  : Xiaofeng
# @FileName: torch[distance_map]
# @Software: Pycharm
# @Usages  :

# -*- coding: utf-8 -*-
# @Time    : 2021/4/3 0003 19:59
# @Author  : Xiaofeng
# @FileName: code[distance_map]
# @Software: Pycharm
# @Usages  :

import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

#ori_path = r'F:\AIM\brain\data\ct'
label_path = r'F:\MD\paper\seg\data\public\resample\gt'
dist_path = r'F:\MD\paper\seg\data\public\resample\dist'

pathExist(dist_path)

def V1():
    for index, file in enumerate(sorted(os.listdir(label_path))):
        if index >= 0:
            print(index,file)
            seg = sitk.ReadImage(os.path.join(label_path, file), sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array[seg_array > 0] = 1

            DM = distance_transform_edt(seg_array).astype(np.float64)

            DM = sitk.GetImageFromArray(DM)

            DM.SetDirection(seg.GetDirection())
            DM.SetOrigin(seg.GetOrigin())
            DM.SetSpacing(seg.GetSpacing())

            sitk.WriteImage(DM, os.path.join(dist_path, file))

V1()