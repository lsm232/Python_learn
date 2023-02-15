from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
import os
import cv2
import bm3d
from skimage.restoration.non_local_means import denoise_nl_means
from skimage.restoration import estimate_sigma


low_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_low'
high_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_high'
save_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\nlm'


files=os.listdir(low_path)
for file in files:
    p=50
    #p=np.random.randint(0,len(files)-1,dtype=np.int)
    low_path_=low_path+'/'+files[p]
    high_path_=high_path+'/'+files[p]

    low_img = Image.open(low_path_).convert('F')
    low_img = np.array(low_img, dtype=np.float32)
    high_img = Image.open(high_path_).convert('F')
    high_img = np.array(high_img, dtype=np.float32)

    #nlm_img=denoise_nl_means(low_img,patch_size=9,patch_distance=9,h=0.1,multichannel=False,fast_mode=False,sigma=0.,preserve_range=True)

    # sigma=estimate_sigma(low_img)
    # nlm_img=denoise_nl_means(low_img,patch_size=7, patch_distance=7, h=0.2,multichannel=False, fast_mode=True)
    # plt.imshow(nlm_img)
    # plt.show()

    bm3d_img=bm3d.bm3d(low_img,sigma_psd=0.32)
    plt.imshow(bm3d_img)
    plt.show()

    psnr=compare_psnr(image_true=high_img,image_test=bm3d_img)
    c=1