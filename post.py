# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 0024 12:14
# @Author  : Xiaofeng
# @FileName: torch[post]
# @Software: Pycharm
# @Usages  :

import copy
import os
import SimpleITK as sitk
import numpy as np
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.io


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


root_dir = r'G:\Public_Dataset\Radiology\3Dircadb\EE-Net\data_resize\seg'
out_dir = r'G:\Public_Dataset\Radiology\3Dircadb\EE-Net\data_resize\seg_post'
pathExist(out_dir)

for idx, file in enumerate(sorted(os.listdir(root_dir))):
    if idx >= 0:
        print('processing label file------------')
        seg = sitk.ReadImage(os.path.join(root_dir, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        ss = np.zeros_like(seg_array)
        seg_copy = copy.deepcopy(seg_array)

        seg1_array = np.zeros_like(seg_array)
        seg2_array = np.zeros_like(seg_array)

        seg1_array[seg_array == 1] = 1
        seg2_array[seg_array == 2] = 1

        # 对门静脉进行后处理
        seg1_array = morphology.binary_dilation(seg1_array)
        seg1_array = morphology.binary_erosion(seg1_array)
        seg1_array = morphology.binary_dilation(seg1_array)
        seg1_array = morphology.binary_erosion(seg1_array)
        seg1_array = morphology.binary_dilation(seg1_array)

        new_seg1 = np.zeros_like(seg1_array)

        # 分别对三个轴面移除细小区域
        for i in range(int(seg1_array.shape[0])):
            # print(i)
            seg_slice = seg1_array[i, :, :]
            new_seg1[i, :, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=5,
                                                                        connectivity=1).astype(seg1_array.dtype)
            new_seg1[i, :, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=200,
                                                                      connectivity=1).astype(seg1_array.dtype)

        for i in range(int(seg1_array.shape[1])):
            # print(i)
            seg_slice = seg1_array[:, i, :]
            new_seg1[:, i, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=5,
                                                                        connectivity=1).astype(seg1_array.dtype)
            new_seg1[:, i, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=200,
                                                                      connectivity=1).astype(seg1_array.dtype)

        for i in range(int(seg1_array.shape[2])):
            # print(i)
            seg_slice = seg1_array[:, :, i]
            new_seg1[:, :, i] = skimage.morphology.remove_small_objects(seg_slice, min_size=5,
                                                                        connectivity=1).astype(seg1_array.dtype)
            new_seg1[:, :, i] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=200,
                                                                      connectivity=1).astype(seg1_array.dtype)

        # save_seg = save_seg * seg_copy

        # 对肝静脉进行后处理
        seg2_array = morphology.binary_dilation(seg2_array)
        seg2_array = morphology.binary_erosion(seg2_array)
        seg2_array = morphology.binary_dilation(seg2_array)
        seg2_array = morphology.binary_erosion(seg2_array)

        new_seg2 = np.zeros_like(seg2_array)

        # 分别对三个轴面移除细小区域
        for i in range(int(seg2_array.shape[0])):
            # print(i)
            seg_slice = seg2_array[i, :, :]
            new_seg2[i, :, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=5,
                                                                        connectivity=1).astype(seg2_array.dtype)
            new_seg2[i, :, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=200,
                                                                      connectivity=1).astype(seg2_array.dtype)

        for i in range(int(seg2_array.shape[1])):
            # print(i)
            seg_slice = seg2_array[:, i, :]
            new_seg2[:, i, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=5,
                                                                        connectivity=1).astype(seg2_array.dtype)
            new_seg2[:, i, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=200,
                                                                      connectivity=1).astype(seg2_array.dtype)

        for i in range(int(seg2_array.shape[2])):
            # print(i)
            seg_slice = seg2_array[:, :, i]
            new_seg2[:, :, i] = skimage.morphology.remove_small_objects(seg_slice, min_size=5,
                                                                        connectivity=1).astype(seg2_array.dtype)
            new_seg2[:, :, i] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=200,
                                                                      connectivity=1).astype(seg2_array.dtype)



        liver_seg1 = morphology.remove_small_objects(new_seg1, 10000, connectivity=1, in_place=True)

        liver_seg1 = morphology.binary_dilation(liver_seg1)
        liver_seg1 = morphology.binary_erosion(liver_seg1)
        liver_seg1 = morphology.binary_dilation(liver_seg1)
        liver_seg1 = morphology.binary_erosion(liver_seg1)
        liver_seg1 = morphology.binary_dilation(liver_seg1)
        liver_seg1 = morphology.binary_erosion(liver_seg1)
        liver_seg1 = morphology.binary_dilation(liver_seg1)
        liver_seg1 = morphology.binary_erosion(liver_seg1)


        liver_seg1 = morphology.remove_small_objects(liver_seg1, 1000, connectivity=1, in_place=True)

        save_seg1 = liver_seg1.astype(np.uint8)


        liver_seg2 = morphology.remove_small_objects(new_seg2, 1000, connectivity=1, in_place=True)

        liver_seg2 = morphology.binary_dilation(liver_seg2)
        liver_seg2 = morphology.binary_erosion(liver_seg2)
        liver_seg2 = morphology.binary_dilation(liver_seg2)
        liver_seg2 = morphology.binary_erosion(liver_seg2)
        liver_seg2 = morphology.binary_dilation(liver_seg2)
        liver_seg2 = morphology.binary_erosion(liver_seg2)
        liver_seg2 = morphology.binary_dilation(liver_seg2)
        liver_seg2 = morphology.binary_erosion(liver_seg2)

        liver_seg2 = morphology.remove_small_objects(liver_seg2, 1000, connectivity=1, in_place=True)

        save_seg2 = liver_seg2.astype(np.uint8)

        ss[save_seg1 == 1] = 1
        ss[save_seg2 == 1] = 2

        save = sitk.GetImageFromArray(ss)

        save.SetDirection(seg.GetDirection())
        save.SetOrigin(seg.GetOrigin())
        save.SetSpacing(seg.GetSpacing())

        sitk.WriteImage(save, os.path.join(out_dir, file))