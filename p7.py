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
from alive_progress import alive_bar


low_path=r'J:\综述\毕业-大论文\第三章传统方法的数据\stage1_test_low'
high_path=r'J:\综述\毕业-大论文\第三章传统方法的数据\stage1_test_high'
save_path=r'J:\综述\毕业-大论文\第三章传统方法的数据\bm3d_matlab'

files=os.listdir(low_path)

def cal_index(test_path,target_path):
    mae=[]
    psnr=[]
    ssim=[]
    for file in files:
        test_path_=test_path+'/'+file
        target_path_=target_path+'/'+file
        test_img=Image.open(test_path_).convert('F')
        target_img=Image.open(target_path_).convert('F')

        test_img=np.array(test_img,dtype=np.float32)
        target_img=np.array(target_img,dtype=np.float32)

        test_img=np.maximum(test_img,0)
        target_img=np.maximum(target_img,0)

        mae+=[compare_mae(y_pred=test_img*4095-1000,y_true=target_img*4095-1000)]
        psnr+=[compare_psnr(image_test=test_img,image_true=target_img)]
        ssim+=[compare_ssim(im1=test_img,im2=target_img)]

    print('psnr_{:.4f}±{:.4f} ssim_{:.4f}±{:.4f} mae_{:.4f}±{:.4f}'.format(np.mean(psnr),np.std(psnr),np.mean(ssim),np.std(ssim),np.mean(mae),np.std(mae)))


cal_index(save_path,high_path)