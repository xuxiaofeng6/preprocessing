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

input_path = r'G:\Hospital\I'
output_path = r'G:\Hospital\I_nii'
pathExist(output_path)

for index,i in enumerate(os.listdir(input_path)):
    if index >= 0:
        mask_path = os.path.join(input_path,i)
        print(mask_path)
        for organ in os.listdir(mask_path):
            if 'StudyInfo' not in organ:
                print(organ)
                organ_path = os.path.join(mask_path,organ)
                print(organ_path)

                reader = sitk.ImageSeriesReader()
                seriesIDs = reader.GetGDCMSeriesIDs(organ_path)
                dcm_series = reader.GetGDCMSeriesFileNames(organ_path,seriesIDs[0])
                reader.SetFileNames(dcm_series)
                img = reader.Execute()

                case_index = i

                out_dir = output_path
                pathExist(out_dir)
                save_name = case_index + '.nii.gz'
                print(save_name)

                sitk.WriteImage(img,os.path.join(out_dir,save_name))

