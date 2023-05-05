import numpy as np
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


orgin_path=r'C:\Users\Zhu\Desktop\ls\80kV_0.7mA_400ms_zhuti-recon.tif'
save_path=r'J:\cyclegan_特征掩码\猪蹄_泛化性测试\norm\0.70mA'

tifs=sitk.ReadImage(orgin_path)
tifs_arr=sitk.GetArrayFromImage(tifs)
i=0
for tif_arr in tifs_arr:
    i=i+1
    #norm 将大于3095的值置为3095，将小于-1000的值置为-1000，再加1000，除以4095
    np.clip(tif_arr,-1000,3095,out=tif_arr)
    tif_arr=((tif_arr+1000)/4095).astype(np.float32)
    """
    plt.imshow(tif_arr)
    plt.show()
    """
    save_name=save_path+'/'+'0.70mA_'+str(i)+'.tif'
    Image.fromarray(tif_arr).save(save_name)



c=1