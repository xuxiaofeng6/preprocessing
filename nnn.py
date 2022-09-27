# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 0026 15:35
# @Author  : Xiaofeng
# @FileName: torch[nnn]
# @Software: Pycharm
# @Usages  :

# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 0026 15:33
# @Author  : Xiaofeng
# @FileName: torch[del.py]
# @Software: Pycharm
# @Usages  :

import os
import shutil

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

root_dir = 'G:\Hospital\ZDH_Brain\Vessel segment'
new_dir = 'G:\Hospital\ZDH_Brain\Vessel segment_'
pathExist(new_dir)

for index,ii in enumerate(sorted(os.listdir(root_dir))):
    if index >= 0:
        print(index, ii)
        case_dir = os.path.join(root_dir,ii)
        new_case_dir = os.path.join(new_dir,ii)
        pathExist(new_case_dir)
        for idx,file in enumerate(sorted(os.listdir(case_dir))):
            #if 'nii.gz' in file:
            path = os.path.join(case_dir,file)
            for ff in os.listdir(path):
                print(ff)
                if '2' in ff:
                    shutil.copyfile(os.path.join(path,ff),os.path.join(new_case_dir,'label2.nii.gz'))
                if '3' in ff:
                    shutil.copyfile(os.path.join(path,ff),os.path.join(new_case_dir,'label3.nii.gz'))



