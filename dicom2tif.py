import numpy as np
from PIL import Image
import os
import pydicom

dicom_path=r'J:\自监督所用数据\挑战赛以外的数据_未归一化\L\L125\1.000000-Low Dose Images-62099'
save_path=r'J:\cyclegan_特征掩码\额外的测试数据\new_low'

files=os.listdir(dicom_path)
for file in files:
    d_path=dicom_path+'/'+file
    dcm=pydicom.read_file(d_path)
    img_arr=dcm.pixel_array
    img=np.asarray(img_arr,dtype=np.float32)/3095

    Image.fromarray(img).save(save_path+'/'+file.split('.')[0]+'.tif')


    c=1