# coding=utf-8
from __future__ import print_function
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

dataDir = r'F:\Gulou_EUS_V2\DL_V3'
out_path = r'F:\Gulou_EUS_V2\DL_V4'

label_dir = os.path.join(dataDir,'label')

for index,idx_name in enumerate(sorted(os.listdir(label_dir))):
    print('index: {} index_name: {}'.format(index, idx_name))
    if index >= 0:

        label_case = os.path.join(label_dir, idx_name)
        print(label_case)

        seg = sitk.ReadImage(label_case, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg).squeeze()
        #print(seg_array.shape)

        i = np.where(seg_array > 0)
        row_min = i[0].min()
        row_max = i[0].max()
        column_min = i[1].min()
        column_max = i[1].max()
        #print(row_min,row_max)
        #print(column_min,column_max)

        image_case = label_case.replace('label','image')
        img = sitk.ReadImage(image_case, sitk.sitkInt16)
        img_array = sitk.GetArrayFromImage(img).squeeze()
        print(img_array.min(),img_array.max())
        #print(img_array.shape)

        cut_image = img_array[row_min:row_max+1,column_min:column_max+1]
        #plt.imshow(cut_image,cmap='gray', vmin=0, vmax=255)
        #plt.show()

        out = os.path.join(out_path,str(seg_array.max()),idx_name.replace('.nii', '.png'))

        plt.imsave(out, cut_image, cmap='gray')


