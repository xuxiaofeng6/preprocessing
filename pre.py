# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 0025 13:39
# @Author  : Xiaofeng
# @FileName: preprocessing[pre]
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

origin_path = r'G:\Public_Dataset\Radiology\MIDAS\test\nii\predict'
tar_path = r'G:\Public_Dataset\Radiology\MIDAS\test\nii\predict_'
pathExist(tar_path)

def remove_small_objects(img):
    binary = copy.copy(img)
    binary[binary > 0] = 1
    labels = morphology.label(binary)
    labels_num = [len(labels[labels == each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    index = list(rank).index(len(rank) - 2)
    new_img = copy.copy(img)
    new_img[labels != index] = 0
    return new_img

def V1():
    for index, file in enumerate(sorted(os.listdir(origin_path))):
        if index >= 0:
            print(index,file)
            seg = sitk.ReadImage(os.path.join(origin_path, file), sitk.sitkUInt8)
            seg_array_origin = sitk.GetArrayFromImage(seg)

            seg_array = morphology.binary_dilation(seg_array_origin)
            #3seg_array = morphology.binary_dilation(seg_array)
            seg_array = morphology.binary_erosion(seg_array)

            new_seg1 = np.zeros_like(seg_array)

            # 分别对三个轴面进行细小孔洞填充
            for i in range(int(seg_array.shape[0])):
                # print(i)
                seg_slice = seg_array[i, :, :]
                # new_seg[i,:,:] = skimage.morphology.remove_small_objects(seg_slice,min_size=10)
                new_seg1[i, :, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=32,
                                                                        connectivity=1).astype(seg_array_origin.dtype)

            for i in range(int(seg_array.shape[1])):
                # print(i)
                seg_slice = new_seg1[:, i, :]
                # new_seg[i,:,:] = skimage.morphology.remove_small_objects(seg_slice,min_size=10)
                new_seg1[:, i, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=32,
                                                                        connectivity=1).astype(seg_array_origin.dtype)

            for i in range(int(seg_array.shape[2])):
                # print(i)
                seg_slice = new_seg1[:, :, i]
                # new_seg[i,:,:] = skimage.morphology.remove_small_objects(seg_slice,min_size=10)
                new_seg1[:, :, i] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=32,
                                                                        connectivity=1).astype(seg_array_origin.dtype)
            
            new_seg2 = np.zeros_like(seg_array)

            
            # 分别对三个轴面移除细小区域
            for i in range(int(seg_array.shape[0])):
                # print(i)
                seg_slice = new_seg1[i, :, :]
                new_seg2[i, :, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=9,
                                                                            connectivity=1).astype(seg_array.dtype)

            for i in range(int(seg_array.shape[1])):
                # print(i)
                seg_slice = new_seg2[:, i, :]
                new_seg2[:, i, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=9,
                                                                            connectivity=1).astype(seg_array.dtype)

            for i in range(int(seg_array.shape[2])):
                # print(i)
                seg_slice = new_seg2[:, :, i]
                new_seg2[:, :, i] = skimage.morphology.remove_small_objects(seg_slice, min_size=9,
                                                                            connectivity=1).astype(seg_array.dtype)

            #new_seg3 = skimage.morphology.remove_small_objects(new_seg2, min_size=1000,
                                                            #connectivity=1).astype(seg_array_origin.dtype)
            
            # 进行连通域提取,移除细小区域,并进行内部的空洞填充
            pred_seg = new_seg2.astype(np.uint8)
            liver_seg = copy.deepcopy(pred_seg)
            liver_seg = measure.label(liver_seg, 4)
            props = measure.regionprops(liver_seg)
            
            max_area = 0
            max_index = 0
            listx = []
            list_index = []
            for index, prop in enumerate(props, start=1):
                #print(index,prop.area)
                #k = zip(str(prop.area),str(index))
                listx.append(prop.area)
                list_index.append(index)
                #if prop.area > max_area:
                    #max_area = prop.area
                    #max_index = index

            zipped = zip(listx,list_index)
            zip_list = list(zipped)
            #print(zip_list)

            zip_list.sort(reverse=True)
            print(zip_list)

            index_num1 = zip_list[0][1]
            index_num2 = zip_list[1][1]
            print(index_num1,index_num2)

            print(liver_seg.shape)

            a = np.where((liver_seg!=index_num1)&(liver_seg!=index_num2))
            liver_seg[a] = 0
            b = np.where((liver_seg == index_num1)|(liver_seg == index_num2))
            liver_seg[b] = 1

            #liver_seg = morphology.binary_erosion(liver_seg)
            #liver_seg = skimage.morphology.remove_small_objects(liver_seg, min_size=1000,
                                                            #connectivity=1).astype(seg_array_origin.dtype)
            
            #print()
            new_seg1 = liver_seg.astype(np.uint8)

            save = sitk.GetImageFromArray(new_seg1)

            save.SetDirection(seg.GetDirection())
            save.SetOrigin(seg.GetOrigin())
            save.SetSpacing(seg.GetSpacing())

            sitk.WriteImage(save, os.path.join(tar_path, file))

def V2():
    for index, file in enumerate(sorted(os.listdir(origin_path))):
        if index >= 0:
            print(index,file)
            seg = sitk.ReadImage(os.path.join(origin_path, file), sitk.sitkUInt8)
            seg_array_origin = sitk.GetArrayFromImage(seg)

            seg_array = morphology.binary_dilation(seg_array_origin)
            seg_array = morphology.binary_erosion(seg_array)

            new_seg1 = np.zeros_like(seg_array)

            # 分别对三个轴面进行细小孔洞填充
            for i in range(int(seg_array.shape[0])):
                # print(i)
                seg_slice = seg_array[i, :, :]
                # new_seg[i,:,:] = skimage.morphology.remove_small_objects(seg_slice,min_size=10)
                new_seg1[i, :, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=64,
                                                                        connectivity=1).astype(seg_array_origin.dtype)

            for i in range(int(seg_array.shape[1])):
                # print(i)
                seg_slice = new_seg1[:, i, :]
                # new_seg[i,:,:] = skimage.morphology.remove_small_objects(seg_slice,min_size=10)
                new_seg1[:, i, :] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=64,
                                                                        connectivity=1).astype(seg_array_origin.dtype)

            for i in range(int(seg_array.shape[2])):
                # print(i)
                seg_slice = new_seg1[:, :, i]
                # new_seg[i,:,:] = skimage.morphology.remove_small_objects(seg_slice,min_size=10)
                new_seg1[:, :, i] = skimage.morphology.remove_small_holes(seg_slice, area_threshold=64,
                                                                        connectivity=1).astype(seg_array_origin.dtype)

            #new_seg2 = np.zeros_like(seg_array)

            """
            # 分别对三个轴面移除细小区域
            for i in range(int(seg_array.shape[0])):
                # print(i)
                seg_slice = new_seg1[i, :, :]
                new_seg2[i, :, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=9,
                                                                            connectivity=1).astype(seg_array.dtype)

            for i in range(int(seg_array.shape[1])):
                # print(i)
                seg_slice = new_seg2[:, i, :]
                new_seg2[:, i, :] = skimage.morphology.remove_small_objects(seg_slice, min_size=9,
                                                                            connectivity=1).astype(seg_array.dtype)

            for i in range(int(seg_array.shape[2])):
                # print(i)
                seg_slice = new_seg2[:, :, i]
                new_seg2[:, :, i] = skimage.morphology.remove_small_objects(seg_slice, min_size=9,
                                                                            connectivity=1).astype(seg_array.dtype)

            #new_seg3 = skimage.morphology.remove_small_objects(new_seg2, min_size=1000,
                                                            #connectivity=1).astype(seg_array_origin.dtype)
            """

            """
            # 进行连通域提取,移除细小区域,并进行内部的空洞填充
            pred_seg = new_seg2.astype(np.uint8)
            liver_seg = copy.deepcopy(pred_seg)
            liver_seg = measure.label(liver_seg, 4)
            props = measure.regionprops(liver_seg)

            max_area = 0
            max_index = 0
            list = []
            list_index = []
            for index, prop in enumerate(props, start=1):
                #print(index,prop.area)
                #k = zip(str(prop.area),str(index))
                list.append(prop.area)
                list_index.append(index)
                if prop.area > max_area:
                    max_area = prop.area
                    max_index = index
            #print(k,type(k))
            print(list)
            print(list_index)
            print(max_index)

            liver_seg[liver_seg != max_index] = 0
            liver_seg[liver_seg == max_index] = 1
            """

            #liver_seg = morphology.binary_erosion()
            liver_seg = skimage.morphology.remove_small_objects(new_seg1, min_size=1000, connectivity=1).astype(seg_array_origin.dtype)

            #liver_seg = liver_seg.astype(np.bool)
            # liver_seg = morphology.remove_small_holes(liver_seg, 5e4 , connectivity=2, in_place=True)
            liver_seg = liver_seg.astype(np.uint8)

            save = sitk.GetImageFromArray(liver_seg)

            save.SetDirection(seg.GetDirection())
            save.SetOrigin(seg.GetOrigin())
            save.SetSpacing(seg.GetSpacing())

            sitk.WriteImage(save, os.path.join(tar_path, file))

V2()