# -*- coding: utf-8 -*-
# @Time    : 2021/11/5 0005 12:42
# @Author  : Xiaofeng
# @FileName: torch[readimg]
# @Software: Pycharm
# @Usages  :
import matplotlib.pyplot as plt
import skimage.io as io
import os
import numpy as np
import plotly.express as px


dir = r'G:\Public_Dataset\BBBC006_v1_images_z_00'
# save_dir = r'G:\Public_Dataset\病理图像\data_glas\Train Folder\labelcol'

for index,i in enumerate(os.listdir(dir)):
    if index <= 10:
        if 'tif' in i:
            img = os.path.join(dir,i)
            img = io.imread(img)

            # img[img==1]=255
            img_arr = np.array(img)
            #
            # io.imsave(os.path.join(dir,i),img_arr)

            # io.imshow(img_arr)
            # plt.show()

            fig = px.imshow(img_arr)
            fig.show()
