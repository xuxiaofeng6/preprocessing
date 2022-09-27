# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 0019 12:27
# @Author  : Xiaofeng
# @FileName: torch[remove]
# @Software: Pycharm
# @Usages  :

#将所有文件重新命名
import os
import shutil
import numpy as np
import pandas as pd

image_dir = r'G:\Hospital\case231_revised\ct_liver'
#label_dir = r'F:\EUS\EUS_all\V3\label'
tar_dir = r'G:\Hospital\case231_revised\liver_all'

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

pathExist(tar_dir)

excel_dir = r'G:\Hospital\case231_revised\sel.xlsx'
data = pd.read_excel(excel_dir)
data_copy = data.copy()

#data = data.set_index('case')

c1 = data.loc[data['level']==0]
c2 = data.loc[data['level']!=0]

c1_dir = os.path.join(tar_dir,'0')
c2_dir = os.path.join(tar_dir,'1')

pathExist(c1_dir)
pathExist(c2_dir)


# print(c1)
# print(c2)

c1_case = c1['case']
c1_array = np.array(c1_case).tolist()
print(c1_array)

c2_case = c2['case']
c2_array = np.array(c2_case).tolist()
print(c2_array)

for index,ii in enumerate(sorted(os.listdir(image_dir))):
    #print(index,ii)
    name = int(str(ii.split('.')[0]))

    if name in c1_array:
        print(name)

        ori = os.path.join(image_dir,ii)
        print(ori)
        shutil.copy(ori, c1_dir)

    if name in c2_array:
        print(name)

        ori = os.path.join(image_dir,ii)
        print(ori)
        shutil.copy(ori, c2_dir)
