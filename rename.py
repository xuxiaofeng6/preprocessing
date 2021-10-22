# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 0029 10:32
# @Author  : Xiaofeng
# @FileName: preprocessing[rename.py]
# @Software: Pycharm
# @Usages  :

#将所有文件重新命名
import os
import re

root_dir = '/data/xiaofeng/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task301_ZDH_vessel/test_label/'

def V1():
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            #num = ii
            #num = ii.split('.')[0]
            #print(num)
            #new_num = "%03d" % int(num)
            #newname = 'ZDH_1'+ new_num +'_0000.nii.gz'
            #newname = 'case50_'+ new_num +'.nii.gz'
            #newname = '1' + new_num +'.nii.gz'
            newname = ii.replace('case50','ZDV')
            print(newname)
            os.rename(os.path.join(root_dir, ii), os.path.join(root_dir, newname))

def V2():
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            newname = str(index+1) +'.nii.gz'
            print(newname)
            #os.rename(os.path.join(root_dir, ii), os.path.join(root_dir, newname))

V1()

def V3():
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            case_dir = os.path.join(root_dir,ii)
            for idx,file in enumerate(sorted(os.listdir(case_dir))):
                if 'label' in file:
                    print('processing label file------------')
                    label_name = re.sub('[\u4e00-\u9fa5]', '', ii) + '-label.nrrd'
                    os.rename(os.path.join(case_dir, file), os.path.join(case_dir, label_name))
                else:
                    print('processing image file------------')
                    img_name = re.sub('[\u4e00-\u9fa5]', '', ii) + '-img.nrrd'
                    os.rename(os.path.join(case_dir, file), os.path.join(case_dir, img_name))

#V3()

def V4():
    #去掉中文字符
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            newname = re.sub('[\u4e00-\u9fa5]', '', ii)
            print(newname)
            os.rename(os.path.join(root_dir,ii),os.path.join(root_dir,newname))

#V4()
