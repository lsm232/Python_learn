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
from scipy.stats import wilcoxon
import csv

a_path=r'J:\综述\毕业-大论文\统计学用表格\第四章\cyclegan_1117_170epoch_ssim.csv'
b_path=r'J:\综述\毕业-大论文\统计学用表格\第四章\cyclegan_1102_155epoch_ssim.csv'



a=[]
b=[]
csv_reader1=csv.reader(open(a_path))
csv_reader2=csv.reader(open(b_path))
for row1 in csv_reader1:
    if row1[0]=='index':
        continue
    else:
        a.append(float(row1[0]))
        c=1
for row2 in csv_reader2:
    if row2[0]=='index':
        continue
    else:
        b.append(float(row2[0]))
        c=1
stat1,p1=shapiro(a)
stat2,p2=shapiro(b)
if (p1>0.05):
    print(a_path,',是正态分布')
if (p2>0.05):
    print(b_path,',是正态分布')

result2=wilcoxon(a,b,correction=True,alternative='two-sided')
print(p1,p2)

print(result2)