# -*- coding: UTF8 -*-

# cPickle是python2系列用的，3系列已经不用了，直接用pickle就好了
import pickle

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
f = open("/data/xiaofeng/nnUNet/nnUNet_raw_data_base/nnUNet_cropped_data/Task301_ZDH_vessel/dataset_properties.pkl",'rb')
data = pickle.load(f)
print(data)
