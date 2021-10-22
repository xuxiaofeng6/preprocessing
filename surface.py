import os
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
import copy

def get_boundaries_of_image(binary_image):
    """
    Return boundaries of binary image
    Parameters
    ----------
    binary_image : 2D or 3D array
        binary image of (m, n) shape to find edges/boundaries of

    Returns
    -------
    boundary_image : 2D OR 3d array
        edges of binary image, has same shape as binary_image

    Notes
    ------
    Uses a erosion based method to find edges of objects in a 2D or 3D image
    """
    sElement = ndimage.generate_binary_structure(binary_image.ndim, 1)
    erode_image = ndimage.morphology.binary_erosion(binary_image, sElement)
    boundary_image = binary_image - erode_image
    return boundary_image

origin_path = '/data/xiaofeng/case161/liver/'
save_path = '/data/xiaofeng/case161/liver_surface'
os.mkdir(save_path)

for index, file in enumerate(sorted(os.listdir(origin_path))):
    if index >= 0:
        print(index,file)
        seg = sitk.ReadImage(os.path.join(origin_path, file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        seg_copy = copy.deepcopy(seg_array)
        seg_array[seg_array > 0] = 1

        #voxel_num1 = np.sum(seg_array)
        #print('origin voxel number:', voxel_num1)

        boundary = get_boundaries_of_image(seg_array)
        boundary = boundary * seg_copy
        #boundary[boundary > 0] = 3

        surface = sitk.GetImageFromArray(boundary)


        surface.SetDirection(seg.GetDirection())
        surface.SetOrigin(seg.GetOrigin())
        surface.SetSpacing(seg.GetSpacing())

        sitk.WriteImage(surface, os.path.join(save_path, file))