# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 0024 21:02
# @Author  : Xiaofeng
# @FileName: preprocessing[folder.py]
# @Software: Pycharm
# @Usages  :

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import re
from time import time
import shutil
from tqdm import tqdm

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

raw_dir = r'G:\Hospital\ZDH_Brain\Test-brainCTA-Zhongda\Test-brainCTA-Zhongda'

Target_dir = r'G:\Hospital\ZDH_Brain\Test-brainCTA-Zhongda\Target'


start = time()
for index, file in enumerate(os.listdir(raw_dir)):
    if index > 21:
        #pathExist(os.path.join(Target_dir,file))
        #s = re.split('(\d+)', file)
        for ff in os.listdir(os.path.join(raw_dir,file)):
            if 'seg.nrrd' in ff:
                #print(ff)
                label_file = ff.split('.nii.seg')[0] + '.nii.gz'
                image_file = re.split('(\d+)', ff.split('.nii.seg')[0])[2] + '.nii.gz'
                #print(label_file)
                #print(image_file)

                src_label = os.path.join(os.path.join(raw_dir, file, label_file))
                #print(src_label)
                target_label = os.path.join(Target_dir, 'gt', file + '.nii.gz')
                print(target_label)
                shutil.copyfile(src_label, target_label)

                src_image = os.path.join(os.path.join(raw_dir, file, image_file))
                #print(src_image)
                target_image = os.path.join(Target_dir, 'ct', file + '.nii.gz')
                print(target_image, target_image)
                shutil.copyfile(src_image, target_image)

