# -*- coding: utf-8 -*-
# @Time    : 2021/12/1 0001 21:35
# @Author  : Xiaofeng
# @FileName: torch[vessel_enhance]
# @Software: Pycharm
# @Usages  :

"""
Sato: https://www.kite.com/python/docs/skimage.filters.sato
"""

import glob
import os
import re
import numpy as np
import SimpleITK as sitk

# ROI bounding box extract lib
from skimage.measure import label
from skimage.measure import regionprops

# Preprocess Vessel
from skimage.filters import frangi, hessian, sato

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
import torch
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Raw_dir = r'J:\Dataset\raw_cut_xyz\image'
# new_dir = r'J:\Dataset\raw_cut_xyz\image_sigmoid_multiply_gaussian'

Raw_dir = r'G:\Hospital\ZDH_Liver\case231_cut_xyz\ct'
new_dir = r'G:\Hospital\ZDH_Liver\case231_cut_xyz\ct_sigmoid_multiply_gaussian'
pathExist(new_dir)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def vessel_sigmoid(x):
    s = 1 / (1 + np.exp((125-x)/25))
    return s

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def VesselEnhance(img, type):
    if type == 'sato':
        filter_img = sato(img, sigmas=range(1, 4, 1), black_ridges=False, mode='constant')
    elif type == 'frangi':
        filter_img = frangi(img, sigmas=range(1, 4, 1), scale_range=None,
                            scale_step=None, alpha=0.5, beta=0.5, gamma=5, black_ridges=False, mode='constant', cval=1)
    return filter_img

start = time()
for file in tqdm(os.listdir(Raw_dir)):
    ct = sitk.ReadImage(os.path.join(Raw_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    # ct_array[ct_array<0] = 0
    # ct_array[ct_array>400] = 400
    #
    # # ct_array = VesselEnhance(ct_array,'sato')
    #
    # ct_array = normalization(ct_array)
    # ct_array = 1 - ct_array
    ct_array = vessel_sigmoid(ct_array)
    ct_array_ = gaussian_filter(ct_array, sigma=1)
    # ct_array = VesselEnhance(ct_array,'sato')c
    ct_array = ct_array * ct_array_

    new_ct = sitk.GetImageFromArray(ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(new_ct, os.path.join(new_dir, file))