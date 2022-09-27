# -*- coding: utf-8 -*-
# @Time    : 2021/6/23 0023 17:44
# @Author  : Xiaofeng
# @FileName: preprocessing[cut_with_mask.py]
# @Software: Pycharm
# @Usages  :

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time
import glob
import re

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage
import nibabel as nib
# ROI bounding box extract lib
from skimage.measure import label
from skimage.measure import regionprops
# Preprocess Vessel
from skimage.filters import frangi, hessian, sato

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def window_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = -150
    MAX_BOUND = 250

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

def cut_with_liver_z():
    raw_path = r'J:\Dataset\Task08_HepaticVessel'
    train_ct_path = os.path.join(raw_path, 'image')
    liver_seg_path = os.path.join(raw_path, 'predict')
    vessel_seg_path = os.path.join(raw_path, 'label')

    new_path = r'J:\Dataset\Task08_HepaticVessel_cut_z'
    new_ct_path = os.path.join(new_path, 'image')
    new_liver_dir = os.path.join(new_path, 'predict')
    new_vessel_dir = os.path.join(new_path, 'label')

    #enhance_ct_path = os.path.join(new_path, 'ct_enhance')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)
    pathExist(new_liver_dir)

    start = time()
    for file in tqdm(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        ct = sitk.ReadImage(os.path.join(train_ct_path, file.replace('3Dircadb1','image')), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)
        liver_array[liver_array != 6] = 0

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file.replace('3Dircadb1','label')), sitk.sitkInt16)
        vessel_array = sitk.GetArrayFromImage(vessel)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(liver_array, axis=(1, 2))
        start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        # # 两个方向上各扩张slice 扩张10
        # start_slice_z = max(0, start_slice_z - 10)
        # end_slice_z = min(liver_array.shape[0] - 1, end_slice_z + 10)

        # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
        # if end_slice_z - start_slice_z + 1 < 32:
        #     print('!!!!!!!!!!!!!!!!')
        #     print(file, 'have too little slice', ct_array.shape[0])
        #     print('!!!!!!!!!!!!!!!!')
        #     continue

        ct_array = ct_array[start_slice_z:end_slice_z + 1,:,:]
        vessel_array = vessel_array[start_slice_z:end_slice_z + 1,:,:]
        liver_array = liver_array[start_slice_z:end_slice_z + 1,:,:]

        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_vessel = sitk.GetImageFromArray(vessel_array)
        new_vessel.SetDirection(ct.GetDirection())
        new_vessel.SetOrigin(ct.GetOrigin())
        new_vessel.SetSpacing(ct.GetSpacing())

        new_liver = sitk.GetImageFromArray(liver_array)
        new_liver.SetDirection(ct.GetDirection())
        new_liver.SetOrigin(ct.GetOrigin())
        new_liver.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))
        sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))

#cut_with_liver_z()

def resize_liver():
    raw_path = r'J:\Dataset\Lung_vessel'
    train_ct_path = os.path.join(raw_path, 'ct')
    vessel_seg_path = os.path.join(raw_path, 'seg')

    new_path = r'J:\Dataset\Lung_vessel_resize'
    new_ct_path = os.path.join(new_path, 'ct')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    #enhance_ct_path = os.path.join(new_path, 'ct_enhance')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)

    start = time()
    for file in tqdm(os.listdir(vessel_seg_path)):
        # 将CT和金标准入读内存
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        # spacing = ct.GetDepth()
        # print(spacing)

        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > 2000] = 2000
        ct_array[ct_array < 1000] = 1000

        ct_array = normalization(ct_array)

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkInt16)
        vessel_array = sitk.GetArrayFromImage(vessel)
        vessel_array[vessel_array != 0] = 1


        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)
        Resized_ct = resize_image_itk(new_ct, (256, 256, 256), resamplemethod=sitk.sitkLinear)
        Resized_ct.SetDirection(ct.GetDirection())
        Resized_ct.SetOrigin(ct.GetOrigin())
        Resized_ct.SetSpacing(ct.GetSpacing())

        new_vessel = sitk.GetImageFromArray(vessel_array)
        #Resized_vessel = resize_image_itk(new_vessel, (256, 256, ct.GetDepth()), resamplemethod=sitk.sitkNearestNeighbor)
        Resized_vessel = resize_image_itk(new_vessel, (256, 256, 256), resamplemethod=sitk.sitkNearestNeighbor)
        Resized_vessel.SetDirection(ct.GetDirection())
        Resized_vessel.SetOrigin(ct.GetOrigin())
        Resized_vessel.SetSpacing(ct.GetSpacing())


        sitk.WriteImage(Resized_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(Resized_vessel, os.path.join(new_vessel_dir, file))

#resize_liver()

def combine():
    raw_path = r'J:\Dataset\raw'
    vessel1_seg_path = os.path.join(raw_path, 'portalvein')
    vessel2_seg_path = os.path.join(raw_path, 'venacava')

    new_vessel_dir = os.path.join(raw_path, 'vessel')

    pathExist(new_vessel_dir)


    start = time()
    for file in tqdm(os.listdir(vessel1_seg_path)):

        vessel1 = sitk.ReadImage(os.path.join(vessel1_seg_path,file), sitk.sitkInt16)
        vessel1_array = sitk.GetArrayFromImage(vessel1)

        vessel2 = sitk.ReadImage(os.path.join(vessel2_seg_path, file.replace('portalvein', 'venacava')), sitk.sitkInt16)
        vessel2_array = sitk.GetArrayFromImage(vessel2)

        new_vessel_array = vessel1_array + vessel2_array

        new_vessel = sitk.GetImageFromArray(new_vessel_array)
        new_vessel.SetDirection(vessel1.GetDirection())
        new_vessel.SetOrigin(vessel1.GetOrigin())
        new_vessel.SetSpacing(vessel1.GetSpacing())

        sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file.replace('portalvein','vessel')))

#combine()

def cut_with_liver_xyz():
    raw_path = r'G:\Hospital\case231_revised'
    ct_path = os.path.join(raw_path, 'ct')
    vessel_seg_path = os.path.join(raw_path, 'gt')

    new_path = r'G:\Hospital\case231_revised_xyz'
    new_ct_dir = os.path.join(new_path, 'ct')
    new_vessel_dir = os.path.join(new_path, 'gt')

    pathExist(new_path)
    pathExist(new_ct_dir)
    pathExist(new_vessel_dir)

    start = time()
    for index,file in enumerate(os.listdir(ct_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
        vessel_array = sitk.GetArrayFromImage(vessel)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        x = np.any(vessel_array, axis=(0, 2))
        start_slice_x, end_slice_x = np.where(x)[0][[0, -1]]

        y = np.any(vessel_array, axis=(0, 1))
        start_slice_y, end_slice_y = np.where(y)[0][[0, -1]]

        z = np.any(vessel_array, axis=(1, 2))
        start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张slice 扩张2
        #start_slice_x = max(0, start_slice_x - 2)
        #end_slice_x = min(liver_array.shape[0] - 1, end_slice_x + 2)

        #start_slice_y = max(0, start_slice_y - 2)
        #end_slice_y = min(liver_array.shape[0] - 1, end_slice_y + 2)

        #start_slice_z = max(0, start_slice_z - 2)
        #end_slice_z = min(liver_array.shape[0] - 1, end_slice_z + 2)

        ct_array = ct_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        vessel_array = vessel_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]

        # 最终将数据保存为nii
        # ct_array = window_normalize(ct_array)
        # # ct_array = standardization(ct_array)
        #
        # ct_array[liver_array != 1] = 0
        # vessel_array[liver_array != 1] = 0

        new_ct = sitk.GetImageFromArray(ct_array)
        #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        # new_liver = sitk.GetImageFromArray(liver_array)
        # #Resized_liver = resize_image_itk(new_liver, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        # new_liver.SetDirection(ct.GetDirection())
        # new_liver.SetOrigin(ct.GetOrigin())
        # new_liver.SetSpacing(ct.GetSpacing())

        # new_portal = sitk.GetImageFromArray(portal_array)
        # #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        # new_portal.SetDirection(ct.GetDirection())
        # new_portal.SetOrigin(ct.GetOrigin())
        # new_portal.SetSpacing(ct.GetSpacing())
        #
        # new_venacava = sitk.GetImageFromArray(venacava_array)
        # #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        # new_venacava.SetDirection(ct.GetDirection())
        # new_venacava.SetOrigin(ct.GetOrigin())
        # new_venacava.SetSpacing(ct.GetSpacing())

        new_vessel = sitk.GetImageFromArray(vessel_array)
        #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        new_vessel.SetDirection(ct.GetDirection())
        new_vessel.SetOrigin(ct.GetOrigin())
        new_vessel.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(new_ct_dir, file))
        #sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))
        # sitk.WriteImage(new_portal, os.path.join(new_portal_dir, file.split('_')[1]))
        # sitk.WriteImage(new_venacava, os.path.join(new_venacava_dir, file.split('_')[1]))
        sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))

#cut_with_liver_xyz()

def select_slice():
    raw_path = r'J:\Dataset\Task08_HepaticVessel_cut_xyz'
    ct_path = os.path.join(raw_path, 'image')
    liver_seg_path = os.path.join(raw_path, 'liver')
    vessel_seg_path = os.path.join(raw_path, 'vessel')

    new_path = r'J:\Dataset\Task08_HepaticVessel_select'
    new_ct_dir = os.path.join(new_path, 'image')
    new_liver_dir = os.path.join(new_path, 'liver')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    pathExist(new_path)
    pathExist(new_ct_dir)
    pathExist(new_liver_dir)
    pathExist(new_vessel_dir)

    start = time()
    for index,file in enumerate(os.listdir(ct_path)):
        ct = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        if ct.GetSpacing()[2] <= 2.5:
            # 将CT和金标准入读内存
            print(index,file,ct.GetSpacing()[2])

            ct_array = sitk.GetArrayFromImage(ct)

            liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
            liver_array = sitk.GetArrayFromImage(liver)

            vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
            vessel_array = sitk.GetArrayFromImage(vessel)


            new_ct = sitk.GetImageFromArray(ct_array)
            #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing(ct.GetSpacing())

            new_liver = sitk.GetImageFromArray(liver_array)
            #Resized_liver = resize_image_itk(new_liver, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
            new_liver.SetDirection(ct.GetDirection())
            new_liver.SetOrigin(ct.GetOrigin())
            new_liver.SetSpacing(ct.GetSpacing())

            new_vessel = sitk.GetImageFromArray(vessel_array)
            #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
            new_vessel.SetDirection(ct.GetDirection())
            new_vessel.SetOrigin(ct.GetOrigin())
            new_vessel.SetSpacing(ct.GetSpacing())

            sitk.WriteImage(new_ct, os.path.join(new_ct_dir, file))
            sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))
            sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))

#select_slice()


def cut_with_xyz():
    raw_path = r'G:\Hospital\norm100\raw'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'gt')

    new_path = r'G:\Hospital\norm100\cut'
    new_ct_path = os.path.join(new_path, 'ct')
    new_liver_dir = os.path.join(new_path, 'gt')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_liver_dir)

    start = time()
    for index,file in enumerate(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file.replace('.nii','_0000.nii')), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)
        liver_array[liver_array != 6] = 0


        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        x = np.any(liver_array, axis=(0, 2))
        start_slice_x, end_slice_x = np.where(x)[0][[0, -1]]

        y = np.any(liver_array, axis=(0, 1))
        start_slice_y, end_slice_y = np.where(y)[0][[0, -1]]

        z = np.any(liver_array, axis=(1, 2))
        start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        ct_array = ct_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        liver_array = liver_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]

        # 最终将数据保存为nii
        #ct_array = standardization(ct_array)
        #ct_array = normalization(ct_array)
        new_ct = sitk.GetImageFromArray(ct_array)
        #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_liver = sitk.GetImageFromArray(liver_array)
        #Resized_liver = resize_image_itk(new_liver, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        new_liver.SetDirection(ct.GetDirection())
        new_liver.SetOrigin(ct.GetOrigin())
        new_liver.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file.replace('.nii','_0000.nii')))
        sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))

cut_with_xyz()

def cut_crop():
    raw_path = r'G:\Public_Dataset\Radiology\3Dircadb\EE-Net\data'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'seg')

    new_path = r'G:\Public_Dataset\Radiology\3Dircadb\EE-Net\data_resize'
    new_ct_path = os.path.join(new_path, 'ct')
    new_seg_dir = os.path.join(new_path, 'seg')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_seg_dir)

    start = time()
    for index,file in enumerate(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file.replace('seg','img')), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        new_spacing = [0.7,0.7,0.7]

        print(ct.GetSpacing())

        print(ct_array.shape)

        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[2]/new_spacing[2], ct.GetSpacing()[1]/new_spacing[1], ct.GetSpacing()[0]/new_spacing[0]), order=3)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[2]/new_spacing[2], ct.GetSpacing()[1]/new_spacing[1], ct.GetSpacing()[0]/new_spacing[0]), order=0)

        print(ct_array.shape)
        print(seg_array.shape)


        # 最终将数据保存为nii
        #ct_array = standardization(ct_array)
        #ct_array = normalization(ct_array)
        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((0.7, 0.7, 0.7))

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((0.7, 0.7, 0.7))

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('seg', 'img')))

#cut_crop()

def onehot(label):
    label1 = np.zeros_like(label)
    label2 = np.zeros_like(label)
    label1[label == 1] = 1
    label2[label == 2] = 1
    label1 = np.expand_dims(label1, axis=0)
    label2 = np.expand_dims(label2, axis=0)
    print(label1.shape)
    print(label2.shape)
    return np.concatenate((label1,label2), axis=0)


def nii2raw():
    raw_path = r'F:\MD\paper\seg\data\public\resample'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'gt')
    mask_path = os.path.join(raw_path,'dist')

    new_path = r'F:\MD\paper\seg\data\public\train'
    new_ct_path = os.path.join(new_path, 'image')
    new_seg_path = os.path.join(new_path, 'label')
    new_mask_path = os.path.join(new_path, 'mask')

    shape_npy = r'F:\MD\paper\seg\data\public\Npy'

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_seg_path)
    pathExist(new_mask_path)
    pathExist(shape_npy)

    for index,file in enumerate(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        # seg_array[seg_array > 0] = 1

        # seg_array =onehot(seg_array)

        mask = sitk.ReadImage(os.path.join(mask_path, file), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(mask)

        # mask_array = onehot(mask_array)

        print(ct_array.shape)
        print(seg_array.shape)
        print(mask_array.shape)

        np.save(os.path.join(shape_npy,file.replace('.nii.gz','.npy')), ct_array.shape)

        ct_array = ct_array.astype(np.float32)
        seg_array = seg_array.astype(np.float32)
        mask_array = mask_array.astype(np.float32)

        ct_array.tofile(os.path.join(new_ct_path,file.replace('.nii.gz','.raw')))
        seg_array.tofile(os.path.join(new_seg_path, file.replace('.nii.gz', '.raw')))
        mask_array.tofile(os.path.join(new_mask_path, file.replace('.nii.gz', '.raw')))


#nii2raw()

def cut_liver():
    raw_path = '/data/xiaofeng/6.2_ZD/'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'liver')

    new_path = '/data/xiaofeng/6.2_ZD_cut_xyz'
    new_ct_path = os.path.join(new_path, 'ct')
    new_liver_dir = os.path.join(new_path, 'liver')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_liver_dir)

    start = time()
    for index,file in enumerate(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file.replace('.nii','_0000.nii')), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        x = np.any(liver_array, axis=(0, 2))
        start_slice_x, end_slice_x = np.where(x)[0][[0, -1]]

        y = np.any(liver_array, axis=(0, 1))
        start_slice_y, end_slice_y = np.where(y)[0][[0, -1]]

        z = np.any(liver_array, axis=(1, 2))
        start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
        if end_slice_z - start_slice_z + 1 < 32:
            print('!!!!!!!!!!!!!!!!')
            print(file, 'have too little slice', ct_array.shape[0])
            print('!!!!!!!!!!!!!!!!')
            continue

        ct_array = ct_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        liver_array = liver_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]

        # 最终将数据保存为nii
        #ct_array = standardization(ct_array)
        #ct_array = normalization(ct_array)
        new_ct = sitk.GetImageFromArray(ct_array)
        #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_liver = sitk.GetImageFromArray(liver_array)
        #Resized_liver = resize_image_itk(new_liver, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        new_liver.SetDirection(ct.GetDirection())
        new_liver.SetOrigin(ct.GetOrigin())
        new_liver.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))

#cut_liver()

def clip_vessel():
    raw_path = r'J:\Dataset\raw_cut_xyz'
    train_ct_path = os.path.join(raw_path, 'image')
    vessel_seg_path = os.path.join(raw_path, 'vessel')

    new_path = r'J:\Dataset\raw_clip_xyz'
    new_ct_path = os.path.join(new_path, 'ct')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)

    start = time()
    for index,file in enumerate(os.listdir(vessel_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        data = ct_array

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
        vessel_array = sitk.GetArrayFromImage(vessel)


        mask = vessel_array > 0  # 生成前景mask
        voxels = list(data[mask][::10])

        voxels_all = []
        voxels_all.append(voxels)

        mean = np.mean(voxels_all)
        std = np.std(voxels_all)
        percentile_99_5 = np.percentile(voxels_all, 99.5)
        percentile_00_5 = np.percentile(voxels_all, 00.5)

        data = np.clip(data, percentile_00_5, percentile_99_5)
        data = (data - mean) / std
        use_nonzero_mask = True
        if use_nonzero_mask:
            data[vessel_array < 0] = 0  # seg<0代表图像里值为0的背景

        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(data)
        #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_vessel = sitk.GetImageFromArray(vessel_array)
        #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        new_vessel.SetDirection(ct.GetDirection())
        new_vessel.SetOrigin(ct.GetOrigin())
        new_vessel.SetSpacing(ct.GetSpacing())


        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))

#clip_vessel()

def crop_with_liver():
        start = time()
        for file in tqdm(os.listdir(train_ct_path)):

            # 将CT和金标准入读内存
            ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            # 将灰度值在阈值之外的截断掉
            #upper = 300
            #lower = 0
            #ct_array[ct_array > upper] = upper
            #ct_array[ct_array < lower] = lower

            liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
            liver_array = sitk.GetArrayFromImage(liver)

            vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
            vessel_array = sitk.GetArrayFromImage(vessel)

            # 找到肝脏区域开始和结束的slice，并各向外扩张slice

            ct_array = ct_array * liver_array
            vessel_array = vessel_array * liver_array
            # 最终将数据保存为nii
            #ct_array = standardization(ct_array)
            #ct_array = normalization(ct_array)
            new_ct = sitk.GetImageFromArray(ct_array)
            #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing(ct.GetSpacing())

            new_vessel = sitk.GetImageFromArray(vessel_array)
            #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
            new_vessel.SetDirection(ct.GetDirection())
            new_vessel.SetOrigin(ct.GetOrigin())
            new_vessel.SetSpacing(ct.GetSpacing())

            new_liver = sitk.GetImageFromArray(liver_array)
            #Resized_liver = resize_image_itk(new_liver, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
            new_liver.SetDirection(ct.GetDirection())
            new_liver.SetOrigin(ct.GetOrigin())
            new_liver.SetSpacing(ct.GetSpacing())

            sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
            sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))
            sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))    

#crop_with_liver()

def VesselEnhance(img, type):
    if type == 'sato':
        filter_img = sato(img, sigmas=range(1, 4, 1), black_ridges=False, mode='constant')
    elif type == 'frangi':
        filter_img = frangi(img, sigmas=range(1, 4, 1), scale_range=None,
                            scale_step=None, alpha=0.5, beta=0.5, gamma=5, black_ridges=False, mode='constant', cval=1)
    return filter_img

def normalize_after_prob(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = np.min(nii_data)
    MAX_BOUND = np.max(nii_data)

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data

def vessel_enhance():
    start = time()
    for file in tqdm(os.listdir(new_ct_path)):

        # 将CT和金标准入读内存
        ct = sitk.ReadImage(os.path.join(new_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        # 将灰度值在阈值之外的截断掉
        #upper = 300
        #lower = 0
        #ct_array[ct_array > upper] = upper
        #ct_array[ct_array < lower] = lower

        # Vessel Enhancement
        ct_array = VesselEnhance(ct_array, type='sato')
        # Normalize
        ct_array = normalize_after_prob(ct_array)
        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)
        #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(enhance_ct_path, file))

#vessel_enhance()

def findidx(file_name):
    # find the idx
    cop = re.compile("[^0-9]")
    idx = cop.sub('', file_name)
    return idx

def combine_vessel_mask(mask1, mask2):
    mask = mask1 + mask2
    mask[mask >= 1] = 1
    return mask

def choose_mask(mask_npy):
    target = [1]
    ix = np.isin(mask_npy, target) # bool array
    # print(ix)
    idx = np.where(ix)
    idx_inv = np.where(~ix) # inverse the bool array
    # print(idx)
    mask_npy[idx] = 1
    mask_npy[idx_inv] = 0
    return mask_npy

def liver_ROI(mask_npy):
    # regionprops tutorial: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
    labeled_img, num = label(mask_npy, return_num=True)
    print(labeled_img.shape)
    print('There are {} regions'.format(num))
    # print(np.max(labeled_img))
    if num > 0:
        regions = regionprops(labeled_img, cache=True)
        for prop in regions:
            box = prop.bbox #Bounding box (min_row, min_col, max_row, max_col)
            area = prop.area #Number of pixels of the region
            ratio = prop.extent #Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
            print(box)
            print(area)
            print(ratio)
            # print(centroid)
            if area >= 800:
                return box

def crop_ROI(npy_data, box):
    xmin, xmax = box[1], box[4]
    ymin, ymax = box[2], box[5]
    zmin, zmax = box[0], box[3]
    # crop to z x 320 x 320
    npy_data_aftercrop = npy_data[zmin:zmax, xmin-5:xmin+315, ymin-5:ymin+315]
    print('crop size:', npy_data_aftercrop.shape)
    if npy_data_aftercrop.shape[1] != 320 or npy_data_aftercrop.shape[2] != 320:
        print('***************************************ERROR CASE*****************************************')
    return npy_data_aftercrop

# val volume h5 generate
def ROI_crop_preprocess(organ='ROI'):

    raw_path = '/data/xiaofeng/3Dircadb/Train Data Cut/'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'liver')
    vessel_seg_path = os.path.join(raw_path, 'vessel')

    new_path = '/data/xiaofeng/3Dircadb/Train Data Cut2'
    new_ct_path = os.path.join(new_path, 'ct')
    new_liver_dir = os.path.join(new_path, 'liver')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    #enhance_ct_path = os.path.join(new_path, 'ct_enhance')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)
    pathExist(new_liver_dir)

    start = time()
    for file in tqdm(os.listdir(train_ct_path)):

        # 将CT和金标准入读内存
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_mask = sitk.GetArrayFromImage(liver)

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
        vessel_mask = sitk.GetArrayFromImage(vessel)

        #image = ct_array.astype(np.float32)
        print('image shape:', ct_array.shape)

        # mask the liver?
        image = liver_mask * ct_array
        vessel_mask = liver_mask * vessel_mask

        # crop the liver area
        # Get the liver area
        #box = liver_ROI(liver_mask)  # (xmin, ymin, zmin, xmax, ymax, zmax)
        # start cropping
        #image = crop_ROI(image, box)
        #mask = crop_ROI(liver_mask, box)
        #vessel_mask = crop_ROI(vessel_mask, box)
            
        # 最终将数据保存为nii
        #ct_array = standardization(ct_array)
        #ct_array = normalization(ct_array)
        new_ct = sitk.GetImageFromArray(image)
        #Resized_ct = resize_image_itk(new_ct, (256,256,128), resamplemethod = sitk.sitkLinear)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_vessel = sitk.GetImageFromArray(vessel_mask)
        #Resized_seg = resize_image_itk(new_seg, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        new_vessel.SetDirection(ct.GetDirection())
        new_vessel.SetOrigin(ct.GetOrigin())
        new_vessel.SetSpacing(ct.GetSpacing())

        new_liver = sitk.GetImageFromArray(liver_mask)
        #Resized_liver = resize_image_itk(new_liver, (256,256,128), resamplemethod= sitk.sitkNearestNeighbor)
        new_liver.SetDirection(ct.GetDirection())
        new_liver.SetOrigin(ct.GetOrigin())
        new_liver.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
        sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))
        sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))   


# Preprocessing Library
def CT_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = -75.0
    MAX_BOUND = 250.0

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data

# Preprocessing Library
def CT_liver_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    MIN_BOUND = -100.0
    MAX_BOUND = 400.0

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data

#ROI_crop_preprocess(organ='ROI')