import copy
import numpy as np
import cv2 as cv
import os
from scipy import ndimage
import skimage.morphology as morphology
from scipy.ndimage import gaussian_filter
import skimage.io
from skimage.measure import label, regionprops
import matplotlib.pyplot
import SimpleITK as sitk

def image_smoothing_gaussian_3d(struct_img, sigma=1, truncate_range=3.0):

    structure_img_smooth = gaussian_filter(struct_img, sigma=sigma, mode='nearest', truncate=truncate_range)

    return structure_img_smooth

#Return boundaries of 2d/3d binary image
def get_boundaries_of_image(binary_image):
    sElement = ndimage.generate_binary_structure(binary_image.ndim, 1)
    erode_image = ndimage.morphology.binary_erosion(binary_image, sElement)
    boundary_image = binary_image - erode_image
    return boundary_image

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

origin_path = r'G:\Public_Dataset\Radiology\ImageCHD\ImageCHD_dataset\data\seg'
skeleton_path = r'G:\Public_Dataset\Radiology\ImageCHD\ImageCHD_dataset\data\seg_'
pathExist(skeleton_path)

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

def pixelcount(regionmask):
    return np.sum(regionmask)

for index,file in enumerate(sorted(os.listdir(origin_path))):
    if index >= 0:
        print(index,file)
        seg = sitk.ReadImage(os.path.join(origin_path, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array[seg_array > 0] = 1
        print(seg_array.ndim)

        #new_seg_array = remove_small_objects(seg_array)
        #new_seg_array = skimage.morphology.remove_small_holes(new_seg_array, area_threshold=512, connectivity=1).astype(new_seg_array.dtype)
        #new_seg_array = ndimage.binary_dilation(new_seg_array).astype(new_seg_array.dtype)
        #parent, tree_traverser = skimage.morphology.max_tree(new_seg_array, connectivity=1)
        #print(type(parent),type(tree_traverser))
        #print(parent.shape, tree_traverser.shape)

        skeleton = image_smoothing_gaussian_3d(seg_array)

        print(pixelcount(skeleton))

        pred_seg = sitk.GetImageFromArray(skeleton)

        pred_seg.SetDirection(seg.GetDirection())
        pred_seg.SetOrigin(seg.GetOrigin())
        pred_seg.SetSpacing(seg.GetSpacing())

        sitk.WriteImage(pred_seg, os.path.join(skeleton_path, file))