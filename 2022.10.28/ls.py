import os
import numpy as np
from PIL import Image
import visdom
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 2022.06.15  cbct论文所用的代码

cbcts_path=r'I:\XiangyaHospital_sCT\testA'
rpcts_path=r'I:\XiangyaHospital_sCT\testB'
models_path=r'I:\XiangyaHospital_sCT\paper7消4\liao_2022.06.04'

cbcts_file=os.listdir(cbcts_path)

cbct_psnr=[]
cbct_ssim=[]
cbct_mae=[]
cbct_rmse=[]
model_psnr=[]
model_ssim=[]
model_mae=[]
model_rmse=[]

for cbct_file in tqdm(cbcts_file):
    #print(cbct_file)

    # read cbct
    cbct_path=cbcts_path+'/'+cbct_file
    cbct_img=np.asarray(Image.open(cbct_path).convert('F'))
    cbct_img=np.maximum(cbct_img,0)
    cbct_img = np.minimum(cbct_img, 1)

    # read rpct
    rpct_path=rpcts_path+'/'+'rpct'+cbct_file.split('.')[0][4:]+'.tif'
    rpct_img = np.asarray(Image.open(rpct_path).convert('F'))
    rpct_img = np.maximum(rpct_img, 0)
    rpct_img = np.minimum(rpct_img, 1)

    # read model
    model_path =models_path+'/'+ 'cbct' + cbct_file.split('.')[0][4:] + '_fake.tif'
    model_img = np.asarray(Image.open(model_path).convert('F'))
    model_img = np.maximum(model_img, 0)
    model_img = np.minimum(model_img, 1)

    # calculate psnr ssim, note: value range [0, 1] is necessary
    cbct_psnr.append(compare_psnr(image_true=rpct_img, image_test=cbct_img))
    cbct_ssim.append(compare_ssim(im1=rpct_img, im2=cbct_img))
    model_psnr.append(compare_psnr(image_true=rpct_img, image_test=model_img))
    model_ssim.append(compare_ssim(im1=rpct_img, im2=model_img))

    # convert to HU
    cbct_img=cbct_img*4095-1000
    rpct_img=rpct_img*4095-1000
    model_img=model_img*4095-1000

    # calculate mae
    cbct_mae.append(compare_mae(y_true=rpct_img, y_pred=cbct_img))
    model_mae.append(compare_mae(y_true=rpct_img, y_pred=model_img))
    cbct_rmse.append(np.sqrt(compare_mse(image0=rpct_img, image1=cbct_img)))
    model_rmse.append(np.sqrt(compare_mse(image0=rpct_img, image1=model_img)))
    # cbct_mse.append(compare_mse(image0=rpct_img, image1=cbct_img))
    # model_mse.append(compare_mse(image0=rpct_img, image1=model_img))


# print results
print('-------------------------------------------------------------------------')
print(cbcts_path)
print(rpcts_path)
print(models_path)
print('cbct: mae {:.4f}±{:.4f}  rmse {:.4f}±{:.4f} psnr {:.4f}±{:.4f}  ssim {:.4f}±{:.4f}'.
      format(np.mean(cbct_mae), np.std(cbct_mae), np.mean(cbct_rmse), np.std(cbct_rmse), np.mean(cbct_psnr), np.std(cbct_psnr), np.mean(cbct_ssim), np.std(cbct_ssim)))
print('model: mae {:.4f}±{:.4f}  rmse {:.4f}±{:.4f} psnr {:.4f}±{:.4f}  ssim {:.4f}±{:.4f}'.
      format(np.mean(model_mae), np.std(model_mae), np.mean(model_rmse), np.std(model_rmse), np.mean(model_psnr), np.std(model_psnr), np.mean(model_ssim), np.std(model_ssim)))

