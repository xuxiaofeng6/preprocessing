# -*- coding: utf-8 -*-
# @Time    : 2021/6/28 0028 9:39
# @Author  : Xiaofeng
# @FileName: preprocessing[move.py]
# @Software: Pycharm
# @Usages  :

import os
import shutil
import tqdm
import nrrd
import numpy as np
import nibabel as nib
import string

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

in_dir = r'G:\Hospital\ZDH_Brain\12-15-CTA\2019-1'
out_dir = r'G:\Hospital\ZDH_Brain\12-15-CTA\2019-1_'
# nii_dir = r'G:\Hospital\ZDH_Brain\Test-brainCTA-Zhongda_729\R\gt_nii'
pathExist(out_dir)
# pathExist(nii_dir)

def V1():
    for file in os.listdir(in_dir):
        print("--------------------------------")
        print(file)
        case_dir = os.path.join(in_dir,file)
        for ff in os.listdir(case_dir):
            if 'nii' in ff:
                print(ff)
                ori_file = os.path.join(case_dir,ff)
                print(ori_file)

                file = file.strip(string.digits)

                new_name = file + '.nii.gz'

                tar_file = os.path.join(out_dir,new_name)
                print(tar_file)
                shutil.copyfile(ori_file,tar_file)

V1()

def V2():
    for file in os.listdir(in_dir):
        case_dir = os.path.join(in_dir,file)
        for ff in os.listdir(case_dir):
            if 'nii.gz' in ff:
                ori_file = os.path.join(case_dir,ff)
                tar_file = os.path.join(out_dir,file + 'nii.gz')
                shutil.copyfile(ori_file,tar_file)

#V2()

def V3():
    for file in os.listdir(in_dir):
        case_dir = os.path.join(in_dir,file)
        for ff in os.listdir(case_dir):
            if 'label.nii.gz' in ff:
                ori_file = os.path.join(case_dir,ff)
                tar_file = os.path.join(out_dir,file + '.nii.gz')
                shutil.copyfile(ori_file,tar_file)

#V3()

def generateAffine(options):
    affine = np.zeros((3, 3))
    for x, i in enumerate(options['space directions']):
        for y, j in enumerate(i):
            if y == 2:
                affine[x, y] = float(j)
            else:
                affine[x, y] = -float(j)

    affine = np.row_stack((affine, np.zeros([1, 3])))
    origin = []
    for x, i in enumerate(options['space origin']):
        if x == 2:
            origin.append(float(i))
        else:
            origin.append(-float(i))
    origin = np.reshape(np.append(origin, 1), (4, 1))
    affine = np.column_stack((affine, origin))

    return affine

def nrrd_nii(nrrd_file):
    label_data, label_options = nrrd.read(nrrd_file)
    label_data[label_data == 6] = 1
    affine = generateAffine(label_options)
    nii_file = nib.Nifti1Image(label_data, affine)
    return nii_file

def VV():
    for index,file in enumerate(os.listdir(out_dir)):
        print(index,"==",file)
        if index >= 0:
            case_dir = os.path.join(out_dir,file)
            nii_file = nrrd_nii(case_dir)
            tar_file = os.path.join(nii_dir,file.split('.')[0] + '.nii.gz')
            nib.save(nii_file,tar_file)

#VV()