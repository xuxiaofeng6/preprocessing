import copy
import numpy as np
import cv2 as cv
import os
from scipy import ndimage
import skimage.morphology as morphology
import skimage.io
from skimage.measure import label, regionprops
import matplotlib.pyplot
import SimpleITK as sitk

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

origin_path = '/data/xiaofeng/6.2_ZD_cut_xyz/pred/'
save_path = '/data/xiaofeng/6.2_ZD_cut_xyz/pred_'
pathExist(save_path)

def remove_small_objects(img):
    binary = copy.copy(img)
    binary[binary > 0] = 1
    labels = morphology.label(binary)
    labels_num = [len(labels[labels == each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    index1 = list(rank).index(len(rank) - 2)
    index2 = list(rank).index(len(rank) - 3)
    #print(len(rank),rank,index)
    new_img = copy.copy(img)
    new_img[np.where((labels != index1)&(labels != index2))] = 0
    return new_img

for index,file in enumerate(sorted(os.listdir(origin_path))):
    if index >= 0:
        print(index,file)
        seg = sitk.ReadImage(os.path.join(origin_path, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        #seg_array[seg_array > 0] = 1
        #print(seg_array.ndim)
        seg_array = remove_small_objects(seg_array)

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(seg.GetDirection())
        new_seg.SetOrigin(seg.GetOrigin())
        new_seg.SetSpacing(seg.GetSpacing())

        sitk.WriteImage(new_seg, os.path.join(save_path, file))