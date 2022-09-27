"""

查看数据轴向spacing分布
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv

ct_path = r'G:\Hospital\ZDH_Liver\case231_cut_xyz\liver'

df = pd.DataFrame(columns=('case', 'volume',))
for file in tqdm(os.listdir(ct_path)):
    print(file)
    seg = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
    seg_array = sitk.GetArrayFromImage(seg)
    #temp = ct.GetSpacing()[-1]
    #$temp = ct.GetSpacing()
    # 计算volume
    spacing = seg.GetSpacing()
    single_voxel_volume = spacing[0] * spacing[1] * spacing[2]
    voxel_num = np.sum(seg_array)
    volume = voxel_num * single_voxel_volume

    #print('-----------------')
    #print(volume)

    row = {'case': file, 'volume': volume, }
    df = df.append(row, ignore_index=True)


print(df.head())
writer = pd.ExcelWriter(os.path.join(ct_path,'spacing.xlsx'),encoding='gbk')
df.to_excel(writer)
writer.save()

