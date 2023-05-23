import numpy as np
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae

path_20=r'J:\cyclegan_特征掩码\猪蹄_泛化性测试\norm\0.20mA'
path_70=r'J:\cyclegan_特征掩码\猪蹄_泛化性测试\norm\0.70mA'
path_35=r'J:\cyclegan_特征掩码\猪蹄_泛化性测试\norm\0.35mA'

files=os.listdir(path_70)

psnr_A=[]
mae_A=[]
ssim_A=[]
psnr_f=[]
mae_f=[]
ssim_f=[]

for i in range(len(files)):
    A=np.asarray(Image.open(path_20+'/0.20mA_'+str(i+1)+'.tif').convert('F'))
    B=np.asarray(Image.open(path_70+'/0.70mA_'+str(i+1)+'.tif').convert('F'))
    #f=np.asarray(Image.open(fake+'/'+file.split('.')[0]+'_fake.tif').convert('F'))  #cyclegan class
    #f=np.asarray(Image.open(fake+'/'+file).convert('F'))  #noise2noise class
    f = np.asarray(Image.open(path_35 + '/0.35mA_' + str(i+1)+'.tif').convert('F'))

    f = np.maximum(f, 0)

    psnr_A.append(compare_psnr(image_true=B,image_test=A))
    ssim_A.append(compare_ssim(im1=B,im2=A))
    mae_A.append(compare_mae(y_true=B*4095-1000,y_pred=A*4095-1000))
    psnr_f.append(compare_psnr(image_true=B, image_test=f))
    ssim_f.append(compare_ssim(im1=B, im2=f))
    mae_f.append(compare_mae(y_true=B * 4095 - 1000, y_pred=f * 4095 - 1000))

print('testA psnr_{:.4f}±{:.4f} ssim_{:.4f}±{:.4f} mae_{:.4f}±{:.4f}'.format(np.mean(psnr_A),np.std(psnr_A),np.mean(ssim_A),np.std(ssim_A),np.mean(mae_A),np.std(mae_A)))
print('fake psnr_{:.4f}±{:.4f} ssim_{:.4f}±{:.4f} mae_{:.4f}±{:.4f}'.format(np.mean(psnr_f),np.std(psnr_f),np.mean(ssim_f),np.std(ssim_f),np.mean(mae_f),np.std(mae_f)))