# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 0003 16:30
# @Author  : Xiaofeng
# @FileName: torch[mm]
# @Software: Pycharm
# @Usages  :

# -*- coding: utf-8 -*-
# @Time    : 2021/6/28 0028 9:39
# @Author  : Xiaofeng
# @FileName: preprocessing[move.py]
# @Software: Pycharm
# @Usages  :

import os
import shutil

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

in_dir = r'G:\Public_Dataset\Radiology\Task888_3Dircadb\3Dircadb_crop2\imagesTr'
out_dir = r'G:\Public_Dataset\Radiology\Task888_3Dircadb\3Dircadb2'
pathExist(out_dir)

def V1():
    for file in os.listdir(in_dir):
        print("--------------------------------")
        print(file)
        img_dir = os.path.join(in_dir,file)
        label_dir = img_dir.replace('imagesTr','labelsTr').replace('img','seg')
        print(img_dir)
        print(label_dir)
        # ori_file = os.path.join(case_dir,ff)
        # tar_file = os.path.join(out_dir,file + '.nrrd')
        tar_folder = os.path.join(out_dir,file.split('_')[0])
        pathExist(tar_folder)
        shutil.copy(img_dir,tar_folder)
        shutil.copy(label_dir, tar_folder)

V1()