from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
import os
import cv2
import bm3d

#这个代码是大论文中第三章涉及的传统滤波算法，包括均值滤波，中值滤波，高斯滤波，双边滤波，NLM，BM3D    注：对于NLM目前只有cv2对其进行封装，且只能处理自然图像uint8，使用matlab官方封装的NLM算法

low_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_low'
high_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_high'
save_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\bm3d'


files=os.listdir(low_path)
for file in files:
    p=np.random.randint(0,len(files)-1,dtype=np.int)
    low_path_=low_path+'/'+files[p]
    high_path_=high_path+'/'+files[p]

    low_img=Image.open(low_path_).convert('F')
    low_img=np.array(low_img,dtype=np.float32)
    high_img = Image.open(high_path_).convert('F')
    high_img = np.array(high_img, dtype=np.float32)

    jun_img=cv2.blur(low_img,ksize=(3,3))
    jun_psnr=compare_psnr(image_true=high_img,image_test=jun_img)
    jun_mae=compare_mae(y_true=high_img,y_pred=jun_img)
    jun_ssim=compare_ssim(im1=high_img,im2=jun_img)

    zhong_img = cv2.medianBlur(low_img,ksize=3)
    zhong_psnr = compare_psnr(image_true=high_img, image_test=zhong_img)
    zhong_mae = compare_mae(y_true=high_img, y_pred=zhong_img)
    zhong_ssim = compare_ssim(im1=high_img, im2=zhong_img)

    gao_img = cv2.GaussianBlur(low_img,ksize=(3,3),sigmaX=0.8,sigmaY=0.8)    #高斯滤波   sigmaX=0.3×[（ksize.width-1）×0.5-1] +0.8
    gao_psnr = compare_psnr(image_true=high_img, image_test=gao_img)
    gao_mae = compare_mae(y_true=high_img, y_pred=gao_img)
    gao_ssim = compare_ssim(im1=high_img, im2=gao_img)

    shuang_img = cv2.bilateralFilter(low_img,7,1,1)  # 高斯滤波   sigmaX=0.3×[（ksize.width-1）×0.5-1] +0.8
    shuang_psnr = compare_psnr(image_true=high_img, image_test=shuang_img)
    shuang_mae = compare_mae(y_true=high_img, y_pred=shuang_img)
    shuang_ssim = compare_ssim(im1=high_img, im2=shuang_img)

    bm_img = bm3d.bm3d(low_img,0.7).astype(np.float32)  # 高斯滤波   sigmaX=0.3×[（ksize.width-1）×0.5-1] +0.8
    bm_psnr = compare_psnr(image_true=high_img, image_test=bm_img)
    bm_mae = compare_mae(y_true=high_img, y_pred=bm_img)
    bm_ssim = compare_ssim(im1=high_img, im2=bm_img)

    print('{:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(jun_psnr,zhong_psnr,gao_psnr,shuang_psnr,bm_psnr))
    print('{:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(jun_mae,zhong_mae,gao_mae,shuang_mae,bm_mae))
    print('{:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(jun_ssim,zhong_ssim,gao_ssim,shuang_ssim,bm_ssim))

    plt.imshow(bm_img,cmap='gray')
    plt.show()

    Image.fromarray(bm_img).save(r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\ls\k.tif')

    c=1