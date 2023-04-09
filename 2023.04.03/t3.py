import csv
import codecs
import numpy as np
import numpy as np
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

#数据存为csv文件
#做u检验

testB=r'J:\cyclegan_特征掩码\dataset\test\high'
fake=r'J:\cyclegan_特征掩码\results\cyclegan_0318-1_175epoch'
save_csv=r'J:\综述\毕业-大论文\统计学用表格\第四章'


files=os.listdir(testB)


psnr_f=[]
mae_f=[]
ssim_f=[]

for file in files:
    B=np.asarray(Image.open(testB+'/'+file).convert('F'))
    f=np.asarray(Image.open(fake+'/'+file.split('.')[0]+'_fake.tif').convert('F'))  #cyclegan class
    #f=np.asarray(Image.open(fake+'/'+file).convert('F'))  #noise2noise class
    f = np.maximum(f, 0)
    psnr_f.append(compare_psnr(image_true=B, image_test=f))
    ssim_f.append(compare_ssim(im1=B, im2=f))
    mae_f.append(compare_mae(y_true=B * 4095 - 1000, y_pred=f * 4095 - 1000))

print('fake psnr_{:.4f}±{:.4f} ssim_{:.4f}±{:.4f} mae_{:.4f}±{:.4f}'.format(np.mean(psnr_f),np.std(psnr_f),np.mean(ssim_f),np.std(ssim_f),np.mean(mae_f),np.std(mae_f)))


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    f=open(file_name,'w')
    f.write('index\n')
    for d in datas:
        f.write(f'{d}\n')
    f.close()
    print("保存文件成功，处理结束")


# data_write_csv(save_csv+'/'+fake.split('\\')[-1]+'_psnr.csv',psnr_f)
# data_write_csv(save_csv+'/'+fake.split('\\')[-1]+'_ssim.csv',ssim_f)
# data_write_csv(save_csv+'/'+fake.split('\\')[-1]+'_mae.csv',mae_f)


