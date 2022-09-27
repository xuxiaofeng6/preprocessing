# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 0026 15:33
# @Author  : Xiaofeng
# @FileName: torch[del.py]
# @Software: Pycharm
# @Usages  :

import os
import shutil

root_dir = 'G:\Hospital\ZDH_Brain\Vessel segment'

for index,ii in enumerate(sorted(os.listdir(root_dir))):
    if index >= 0:
        print(index, ii)
        case_dir = os.path.join(root_dir,ii)
        for idx,file in enumerate(sorted(os.listdir(case_dir))):
            if 'nii.gz' in file:
                path = os.path.join(case_dir,file)
                os.remove(path)
                #shutil.rmtree(path)


