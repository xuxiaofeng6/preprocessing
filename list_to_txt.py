# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 0003 12:47
# @Author  : Xiaofeng
# @FileName: torch[list_tp_txt.py]
# @Software: Pycharm
# @Usages  :
import math
import os
import random

path = r'J:\Dataset\SegWithDistMap'
h5_path = os.path.join(path,'h5')
total_num = 110

name_list = []

for i in os.listdir(h5_path):
    name =i
    name_list.append(name)

random.shuffle(name_list)
print(name_list)

len_list = len(name_list)

print(len_list)

train_list = name_list[0:math.ceil(total_num*0.7)]
test_list = name_list[math.ceil(total_num*0.7):]
print(len(train_list))
print(len(test_list))

with open(os.path.join(path,"train.txt"), 'w') as output:
    for row in train_list:
        output.write(str(row) + '\n')

with open(os.path.join(path,"test.txt"), 'w') as output:
    for row in test_list:
        output.write(str(row) + '\n')