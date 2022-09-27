# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 0029 10:32
# @Author  : Xiaofeng
# @FileName: preprocessing[rename.py]
# @Software: Pycharm
# @Usages  :

#将所有文件重新命名
import os
import re
import pandas as pd

gt_dir = r'G:\Hospital\I_nii'
# pred_dir = r'G:\Hospital\I_nii_new_name'

def V():
    df = pd.DataFrame(columns=('GT', 'Pred',))
    for index,ii in enumerate(sorted(os.listdir(gt_dir))):
        if index >= 0:
            print(index, ii)
            #num = ii

            gt_ = os.path.join(gt_dir,ii)
            pred_ = os.path.join(pred_dir,ii)

            row = {'GT': gt_, 'Pred': pred_, }
            df = df.append(row, ignore_index=True)

    print(df.head())
    writer = pd.ExcelWriter(r'I:\Task302_ZDH_vessel\dice.xlsx')
    df.to_excel(writer)
    writer.save()

#V()

def V0():
    df = pd.DataFrame(columns=('origin_name', 'new_name',))
    for index,ii in enumerate(sorted(os.listdir(gt_dir))):
        if index >= 0:
            print(index, ii)
            #num = ii


            origin_name = ii
            print(origin_name)

            newname = str(index+1)  + '.nii.gz'
            row = {'origin_name': ii, 'new_name': newname, }
            df = df.append(row, ignore_index=True)
            #
            print(newname)
            os.rename(os.path.join(gt_dir, ii), os.path.join(gt_dir, newname))

    print(df.head())
    writer = pd.ExcelWriter(r'G:\Hospital\I_nii\name.xlsx')
    df.to_excel(writer)
    writer.save()

#V0()

def V1():
    root_dir = r'G:\Hospital\I_nii'
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            #num = ii
            #newname = ii.split('.')[0].split('_')[1] + '.nii.gz'
            #newname = ii.split('_')[1] + '.nii.gz'
            #print(num)
            new_num = "%03d" % int(ii.split('.nii')[0])
            newname = 'ZDH_1'+ new_num +'_0000.nii.gz'
            #
            #newname = '1' + new_num +'.nii.gz'
            # newname = ii.replace('case50','ZDV')
            #print(newname)

            #newname = str(new_num)  + '.nii.gz'
            #
            print(newname)
            os.rename(os.path.join(root_dir, ii), os.path.join(root_dir, newname))

V1()

def V2():
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            newname = str(index+1) +'.nii.gz'
            print(newname)
            #os.rename(os.path.join(root_dir, ii), os.path.join(root_dir, newname))

#V1()

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
