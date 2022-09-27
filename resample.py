# coding=utf-8
import nibabel as nib
import os
from dipy.align.reslice import reslice
import scipy.io as sio
import numpy as np


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

ct_path = r'F:\MD\paper\seg\data\public\raw\ct'
gt_path = r'F:\MD\paper\seg\data\public\raw\gt'
outpath = r'F:\MD\paper\seg\data\public\raw_resample'
out_ct_path = os.path.join(outpath,'ct')
out_gt_path = os.path.join(outpath,'gt')
pathExist(out_ct_path)
pathExist(out_gt_path)

for index,idx_name in enumerate(sorted(os.listdir(ct_path))):
    if index >= 0:
        print('index: {} index_name: {}'.format(index, idx_name))
        nii_ct = nib.load(os.path.join(ct_path, idx_name))
        nii_label = nib.load(os.path.join(gt_path, idx_name))
        affine = nii_ct.affine
        zoom = nii_ct.header.get_zooms()[:3]
        img = nii_ct.get_data()
        label = nii_label.get_data()
        new_img, affine = reslice(img, affine, zoom, (1, 1, 1))
        new_label, affine = reslice(label, affine, zoom, (1, 1, 1))
        nib.save(nib.Nifti1Image(new_img, affine), os.path.join(out_ct_path,idx_name))
        nib.save(nib.Nifti1Image(new_label, affine), os.path.join(out_gt_path, idx_name))
