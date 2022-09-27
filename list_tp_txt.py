# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 0003 12:47
# @Author  : Xiaofeng
# @FileName: torch[list_tp_txt.py]
# @Software: Pycharm
# @Usages  :
import math
import os

path = r'G:\Hospital\ZDH_Liver\case50_crop_xyz_patch_h5\ct'

name_list = []

for i in os.listdir(path):
    name =i.split('.')[0]
    name_list.append(name)

print(name_list)

len_list = len(name_list)

print(len_list)

train_list = name_list[0:math.ceil(1296*0.6)]
val_list = name_list[math.ceil(1296*0.6):math.ceil(1296*0.8)]
test_list = name_list[math.ceil(1296*0.8):]
print(len(train_list))
print(len(val_list))
print(len(test_list))

with open(os.path.join(path,"train.txt"), 'w') as output:
    for row in train_list:
        output.write(str(row) + '\n')

with open(os.path.join(path,"val.txt"), 'w') as output:
    for row in val_list:
        output.write(str(row) + '\n')

with open(os.path.join(path,"test.txt"), 'w') as output:
    for row in test_list:
        output.write(str(row) + '\n')