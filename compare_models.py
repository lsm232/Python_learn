import numpy as np
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
import matplotlib.pyplot as plt

target_path=r'J:\cyclegan_特征掩码\dataset\test\high'
model1_path=r'C:\Users\Zhu\Desktop\pl\lp2'
model2_path=r'J:\cyclegan_特征掩码\results\cyclegan_1102_155epoch'

files=os.listdir(target_path)
for i in range(len(files)):
    file=files[i+200]
    target_img=np.asarray(Image.open(target_path+'/'+file).convert('F'))
    model1_img = np.asarray(Image.open(model1_path + '/' + file.split('.')[0] + '_fake.tif').convert('F'))
    model2_img = np.asarray(Image.open(model2_path + '/' + file.split('.')[0] + '_fake.tif').convert('F'))

    target_img=np.maximum(target_img,0)
    model1_img=np.maximum(model1_img,0)
    model2_img=np.maximum(model2_img,0)

    res1=target_img-model1_img
    res2=target_img-model2_img

    cm1=plt.cm.get_cmap('bwr')

    res1=np.abs(res1)
    res2=np.abs(res2)

    plt.subplot(1,2,1)
    c1=plt.imshow(res1,cmap=cm1,vmin=-0.3,vmax=0.3)
    plt.colorbar(c1)
    plt.subplot(1, 2, 2)
    c2 = plt.imshow(res2, cmap=cm1, vmin=-0.3, vmax=0.3)
    plt.colorbar(c2)
    plt.show()
    c=1