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

dict_methods={'0':'均值滤波','1':'中值滤波','2':'高斯滤波','3':'双边滤波','4':'BM3D'}
files = os.listdir(low_path)

#
def denosing(method=''):

    if '均值滤波'==dict_methods[method]:
        print('你选择的去噪算法是:均值滤波')
        for file in files:
            path = low_path + '\\' + file
            img = Image.open(path).convert('F')
            img=np.array(img,dtype=np.float32)
            out=cv2.blur(img,ksize=(3,3))
            save_path2 = save_path + '/' + file
            Image.fromarray(out).save(save_path2)
            c = 1
        cal_index(save_path, high_path)

    elif '中值滤波'==dict_methods[method]:
        print('你选择的去噪算法是:中值滤波')
        for file in files:
            path = low_path + '\\' + file
            img = Image.open(path).convert('F')
            img = np.array(img, dtype=np.float32)
            out=cv2.medianBlur(img,ksize=3)
            save_path2 = save_path + '/' + file
            Image.fromarray(out).save(save_path2)
            c = 1
        cal_index(save_path, high_path)

    elif '高斯滤波' == dict_methods[method]:
        print('你选择的去噪算法是:高斯滤波')
        for file in files:
            path = low_path + '\\' + file
            img = Image.open(path).convert('F')
            img = np.array(img, dtype=np.float32)
            out = cv2.GaussianBlur(img,ksize=(3,3),sigmaX=0.8,sigmaY=0.8)    #高斯滤波   sigmaX=0.3×[（ksize.width-1）×0.5-1] +0.8
            save_path2 = save_path + '/' + file
            Image.fromarray(out).save(save_path2)
            c = 1
        cal_index(save_path, high_path)

    elif '双边滤波' == dict_methods[method]:
        print('你选择的去噪算法是:双边滤波')
        for file in files:
            path = low_path + '\\' + file
            img = Image.open(path).convert('F')
            img = np.array(img, dtype=np.float32)
            out=cv2.bilateralFilter(img,7,1,1)
            save_path2 = save_path + '/' + file
            Image.fromarray(out).save(save_path2)
            c = 1
        cal_index(save_path, high_path)

    elif 'BM3D' == dict_methods[method]:
        print('你选择的去噪算法是:BM3D')
        for file in files:
            path = low_path + '\\' + file
            img = Image.open(path).convert('F')
            img = np.array(img, dtype=np.float32)
            out=bm3d.bm3d(img,0.01)
            #
            # plt.imshow(out,cmap='gray')
            # plt.show()
            # c=1

            #
            save_path2 = save_path + '/' + file
            Image.fromarray(out).save(save_path2)
            c = 1
        cal_index(save_path, high_path)

    else:
        pass


def cal_index(test_path,target_path):
    mae=[]
    psnr=[]
    ssim=[]
    for file in files:
        test_path_=test_path+'/'+file
        target_path_=target_path+'/'+file
        test_img=Image.open(test_path_).convert('F')
        target_img=Image.open(target_path_).convert('F')

        test_img=np.maximum(test_img,0)
        target_img=np.maximum(target_img,0)

        mae+=[compare_mae(y_pred=test_img*4095-1000,y_true=target_img*4095-1000)]
        psnr+=[compare_psnr(image_test=test_img,image_true=target_img)]
        ssim+=[compare_ssim(im1=test_img,im2=target_img)]

    print('psnr_{:.4f}±{:.4f} ssim_{:.4f}±{:.4f} mae_{:.4f}±{:.4f}'.format(np.mean(psnr),np.std(psnr),np.mean(ssim),np.std(ssim),np.mean(mae),np.std(mae)))


denosing('4')
