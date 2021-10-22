from py_SAINT.STAGE1 import nii2pickle
from py_SAINT.STAGE1 import interpolation

def step1():
    ori_dir_path = '/data/xiaofeng/SAINT/file_nii/'
    output_file_path = '/data/xiaofeng/SAINT/Data/Stage1_Input/TEST/HR/'

    nii2pickle.nii2pt(ori_dir_path,output_file_path)

def step2():
    dir_data = '/data/xiaofeng/SAINT/Data/Stage1_Input/'
    save = '/data/xiaofeng/SAINT/Data/Stage1_output_sag_cor/'


    interpolation.get_Stage1_result (scale ='4',save = save,dir_data = dir_data,n_colors =3 ,n_GPUs =1,rgb_range =4000, view ='sag',gpu='0')

    interpolation.get_Stage1_result (scale ='4',save = save,dir_data = dir_data,n_colors =3 ,n_GPUs =1,rgb_range =4000, view ='cor',gpu='0')

step2()