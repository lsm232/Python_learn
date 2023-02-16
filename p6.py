from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
import os
import cv2
import bm3d
from skimage.restoration import estimate_sigma
from skimage.restoration.non_local_means import denoise_nl_means
import tqdm

low_path=r'J:\综述\毕业-大论文\第三章传统方法的数据\stage1_test_low'
high_path=r'J:\综述\毕业-大论文\第三章传统方法的数据\stage1_test_high'


psnr_=[]
ssim_=[]
mae_=[]

files=os.listdir(high_path)
for file in tqdm.tqdm(files):
    l_path=low_path+'/'+file
    h_path=high_path+'/'+file
    low_img=Image.open(l_path).convert('F')
    high_img=Image.open(h_path).convert('F')

    low_img=np.array(low_img,dtype=np.float32)
    high_img=np.array(high_img,dtype=np.float32)

    #out = cv2.bilateralFilter(low_img,3,1.30,1.30)  #高斯滤波   sigmaX=0.3×[（ksize.width-1）×0.5-1] +0.8
    out=bm3d.bm3d(low_img,0.31)
    out=np.array(out,dtype=np.float32)

    psnr_+=[compare_psnr(image_true=high_img,image_test=out)]
    ssim_+=[compare_ssim(im1=high_img,im2=out)]
    mae_+=[compare_mae(y_true=high_img*4095-1000,y_pred=out*4095-1000)]

print('psnr {:.4f} {:.4f};ssim {:.4f} {:.4f};mae {:.4f} {:.4f}'.format(np.mean(psnr_),np.std(psnr_),np.mean(ssim_),np.std(ssim_),np.mean(mae_),np.std(mae_)))


