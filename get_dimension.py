"""

查看数据轴向spacing分布
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from tqdm import tqdm
import SimpleITK as sitk

import pandas as pd
import csv

spacing_list = []
ct_path = '/data/xiaofeng/case231_cut_z/ct/'

for file in tqdm(os.listdir(ct_path)):
    print(file)
    ct = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
    #temp = ct.GetSpacing()[-1]
    temp = sitk.GetArrayFromImage(ct).shape

    print('-----------------')
    print(temp)

    spacing_list.append(temp)

#print('mean:', sum(spacing_list) / len(spacing_list))
#print('max:', max(spacing_list))
#print('min:', min(spacing_list))

spacing_list.sort()
print(spacing_list)

save=pd.DataFrame(data=spacing_list)
#save.to_csv('spacing.csv',encoding='gbk')
save.to_csv(os.path.join(ct_path,'dimension.csv'),encoding='gbk')

# 训练集中的平均spacing是1.59mm
# 测试集中的数据的spacing都是1mm
