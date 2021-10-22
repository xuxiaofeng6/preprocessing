"""
import nrrd # pip install pynrrd
import nibabel as nib # pip install nibabel
import numpy as np

# load nrrd 
_nrrd = nrrd.read('/data/xiaofeng/Brain_yu/45钟子威/45钟子威-img.nrrd')
data = _nrrd[0]
header = _nrrd[1]
print(data.shape, header)

# save nifti
img = nib.Nifti1Image(data, np.eye(4))
nib.save(img,'/data/xiaofeng/Brain_yu/45钟子威/mri.nii.gz')
"""

import nrrd
import nibabel as nib
import os
import sys

root_dir = '/data/xiaofeng/Brain_yu'
save_dir = root_dir + '_nii'
print(save_dir)

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def V():
    for filename in os.listdir(dir):
        if filename.endswith(".nrrd"):
            nrrd_name = os.path.join(dir,filename)
            print(nrrd_name)
            data, header = nrrd.read(nrrd_name)
            save_image = nib.Nifti1Image(dataobj = data, affine=None)
            nib.save(save_image, nrrd_name.split('.')[0] + '.nii.gz')

def nrrd2nii(nrrd_file):
    data, header = nrrd.read(nrrd_file)
    nii_file = nib.Nifti1Image(dataobj = data, affine=None)
    return nii_file

def V1():
    for index,ii in enumerate(sorted(os.listdir(root_dir))):
        if index >= 0:
            print(index, ii)
            case_dir = os.path.join(root_dir,ii)
            for idx,file in enumerate(sorted(os.listdir(case_dir))):
                print(idx,file)
                save_image = nrrd2nii(os.path.join(case_dir,file))
                pathExist(os.path.join(save_dir,ii))
                nib.save(save_image, os.path.join(save_dir,ii,file.replace('.nrrd','.nii.gz')))

V1()