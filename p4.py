from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae

low_img=np.array(Image.open(r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_low\C052-001.tif').convert('F'),dtype=np.float32)
high_img=np.array(Image.open(r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_high\C052-001.tif').convert('F'),dtype=np.float32)
p=compare_psnr(high_img,low_img)
m=compare_mae(high_img,low_img)
s=compare_ssim(high_img,low_img)
c=1

