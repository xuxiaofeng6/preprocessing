# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 0030 13:11
# @Author  : Xiaofeng
# @FileName: torch[dcm2nii]
# @Software: Pycharm
# @Usages  :
import os
import SimpleITK as sitk

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

input_path = r'G:\Public_Dataset\Radiology\MIDAS\test\mha'
output_path = r'G:\Public_Dataset\Radiology\MIDAS\test\nii'
pathExist(output_path)

for i in os.listdir(input_path):
    print(i)
    ct = sitk.ReadImage(os.path.join(input_path, i), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing(ct.GetSpacing())
    #
    sitk.WriteImage(new_ct,os.path.join(output_path,i.replace('mha','nii.gz')))

