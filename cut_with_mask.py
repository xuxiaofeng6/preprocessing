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
    raw_path = '/data/xiaofeng/case231'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'liver')
    vessel_seg_path = os.path.join(raw_path, 'vessel')

    new_path = '/data/xiaofeng/case231_cut_z/'
    new_ct_path = os.path.join(new_path, 'ct')
    new_liver_dir = os.path.join(new_path, 'liver')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    #enhance_ct_path = os.path.join(new_path, 'ct_enhance')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)
    pathExist(new_liver_dir)

    start = time()
    for file in tqdm(os.listdir(vessel_seg_path)):
        # 将CT和金标准入读内存
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)
        liver_array[liver_array > 1] = 1

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
        vessel_array = sitk.GetArrayFromImage(vessel)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(liver_array, axis=(1, 2))
        start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张slice 扩张2
        start_slice_z = max(0, start_slice_z - 2)
        end_slice_z = min(liver_array.shape[0] - 1, end_slice_z + 2)

        # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
        if end_slice_z - start_slice_z + 1 < 32:
            print('!!!!!!!!!!!!!!!!')
            print(file, 'have too little slice', ct_array.shape[0])
            print('!!!!!!!!!!!!!!!!')
            continue

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

def cut_with_liver_xyz():
    raw_path = '/data/xiaofeng/case50/'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'liver')
    vessel_seg_path = os.path.join(raw_path, 'vessel')

    new_path = '/data/xiaofeng/case50_cut_xyz'
    new_ct_path = os.path.join(new_path, 'ct')
    new_liver_dir = os.path.join(new_path, 'liver')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)
    pathExist(new_liver_dir)

    start = time()
    for index,file in enumerate(os.listdir(liver_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file.replace('.nii','_0000.nii')), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
        vessel_array = sitk.GetArrayFromImage(vessel)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        x = np.any(liver_array, axis=(0, 2))
        start_slice_x, end_slice_x = np.where(x)[0][[0, -1]]

        y = np.any(liver_array, axis=(0, 1))
        start_slice_y, end_slice_y = np.where(y)[0][[0, -1]]

        z = np.any(liver_array, axis=(1, 2))
        start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张slice 扩张2
        #start_slice_x = max(0, start_slice_x - 2)
        #end_slice_x = min(liver_array.shape[0] - 1, end_slice_x + 2)

        #start_slice_y = max(0, start_slice_y - 2)
        #end_slice_y = min(liver_array.shape[0] - 1, end_slice_y + 2)

        #start_slice_z = max(0, start_slice_z - 2)
        #end_slice_z = min(liver_array.shape[0] - 1, end_slice_z + 2)

        # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
        #if end_slice_z - start_slice_z + 1 < 32:
            #print('!!!!!!!!!!!!!!!!')
            #print(file, 'have too little slice', ct_array.shape[0])
            #print('!!!!!!!!!!!!!!!!')
            #continue

        ct_array = ct_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        vessel_array = vessel_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        liver_array = liver_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]

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

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, file.replace('.nii','_0000.nii')))
        sitk.WriteImage(new_vessel, os.path.join(new_vessel_dir, file))
        sitk.WriteImage(new_liver, os.path.join(new_liver_dir,file))

cut_with_liver_xyz()

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

def cut_with_liver_region():
    raw_path = '/data/xiaofeng/3Dircadb/Train Data Cut xyz/'
    train_ct_path = os.path.join(raw_path, 'ct')
    liver_seg_path = os.path.join(raw_path, 'liver')
    vessel_seg_path = os.path.join(raw_path, 'vessel')

    new_path = '/data/xiaofeng/3Dircadb/Train Data Cut region_'
    new_ct_path = os.path.join(new_path, 'ct')
    new_liver_dir = os.path.join(new_path, 'liver')
    new_vessel_dir = os.path.join(new_path, 'vessel')

    pathExist(new_path)
    pathExist(new_ct_path)
    pathExist(new_vessel_dir)
    pathExist(new_liver_dir)

    start = time()
    for index,file in enumerate(os.listdir(vessel_seg_path)):
        # 将CT和金标准入读内存
        print(index,file)
        ct = sitk.ReadImage(os.path.join(train_ct_path, file), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        liver = sitk.ReadImage(os.path.join(liver_seg_path, file), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)

        vessel = sitk.ReadImage(os.path.join(vessel_seg_path, file), sitk.sitkUInt8)
        vessel_array = sitk.GetArrayFromImage(vessel)

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        #x = np.any(liver_array, axis=(0, 2))
        #start_slice_x, end_slice_x = np.where(x)[0][[0, -1]]

        #y = np.any(liver_array, axis=(0, 1))
        #start_slice_y, end_slice_y = np.where(y)[0][[0, -1]]

        #z = np.any(liver_array, axis=(1, 2))
        #start_slice_z, end_slice_z = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张slice 扩张2
        #start_slice_x = max(0, start_slice_x - 2)
        #end_slice_x = min(liver_array.shape[0] - 1, end_slice_x + 2)

        #start_slice_y = max(0, start_slice_y - 2)
        #end_slice_y = min(liver_array.shape[0] - 1, end_slice_y + 2)

        #start_slice_z = max(0, start_slice_z - 2)
        #end_slice_z = min(liver_array.shape[0] - 1, end_slice_z + 2)

        # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
        #if end_slice_z - start_slice_z + 1 < 32:
            #print('!!!!!!!!!!!!!!!!')
            #print(file, 'have too little slice', ct_array.shape[0])
            #print('!!!!!!!!!!!!!!!!')
            #continue

        #ct_array = ct_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        #vessel_array = vessel_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]
        #liver_array = liver_array[start_slice_z:end_slice_z + 1, start_slice_x:end_slice_x + 1, start_slice_y:end_slice_y + 1]

        #ct_array = ct_array * liver_array
        #vessel_array = vessel_array * liver_array

        ct_array[liver_array != 1] = 0
        vessel_array[liver_array != 1] = 0

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

#cut_with_liver_region()

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