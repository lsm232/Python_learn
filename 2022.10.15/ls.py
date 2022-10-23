import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import numpy as np
import seaborn as sns


def draw_histogram_and_line_single(img_path:str,color:str,bins:int,value:int,alpha:float,label:str):
    path = img_path
    files = os.listdir(path)
    all_imgs = torch.rand(size=(len(files), 512, 512))

    for i, file in enumerate(files):
        img_path = os.path.join(path, file)
        img = Image.open(img_path).convert("F")
        img = np.asarray(img)
        img_ct = img * 4095 - 1000
        img_ct = img_ct.astype(np.int32)

        img_ct = torch.from_numpy(img_ct)
        all_imgs[i, :, :] = img_ct

    all_imgs = np.asarray(all_imgs).reshape(-1)
    result = np.histogram(all_imgs, bins=bins, range=value)
    plt.hist(x=all_imgs, bins=bins, range=value,facecolor=color,alpha=alpha,edgecolor=color)
    a = np.linspace(value[0], value[1], bins)
    plt.plot(a, result[0],color=color,label=label)



def draw_histogram_and_line_multi(path,color,bins,alpha,value):
    """params
    path=[cbct_path,rpct_path,model1_path,model2_path...]
    color=[cbct_color,rpct-color,model1-color...]
    bins=20
    alpha=0.8
    ct_range=[-value,value]
    """
    assert len(path)==5,"需要依次传入cbct,rpct,cyclegan,res-cyclegan,trans-cyclegan的路径"
    assert len(color)==5,"需要依次传入cbct,rpct,cyclegan,res-cyclegan,trans-cyclegan绘图对应的颜色"
    labels=['cbct','rpct','cyclegan','respath-cyclegan','trans-cyclegan']
    draw_histogram_and_line_single(path[0], color[0], bins, value, alpha=alpha,label=labels[0])
    draw_histogram_and_line_single(path[1], color[1], bins, value, alpha=alpha,label=labels[1])
    draw_histogram_and_line_single(path[2], color[2], bins, value, alpha=alpha,label=labels[2])
    draw_histogram_and_line_single(path[3], color[3], bins, value, alpha=alpha,label=labels[3])
    draw_histogram_and_line_single(path[4], color[4], bins, value, alpha=alpha,label=labels[4])
    plt.xlabel("HU")
    plt.ylabel("number")
    plt.legend()
    plt.show()


paths=[r'F:\histogram_\ls\cbct',
       r'F:\histogram_\ls\rpct',
       r'F:\histogram_\ls\cyclegan',
       r'F:\histogram_\ls\respath',
       r'F:\histogram_\ls\trans'
       ] #需要依次传入cbct,rpct,cyclegan,res-cyclegan,trans-cyclegan对应的路径
#https://blog.csdn.net/qq_45269147/article/details/105732597  颜色选择
colors=['black','blue','green','yellow','red']  #需要依次传入cbct,rpct,cyclegan,res-cyclegan,trans-cyclegan绘图对应的颜色
bins=15  #多少列
value=[-500,500]  #灰度值范围
alpha=0.  #柱状图的透明度

draw_histogram_and_line_multi(paths,colors,bins,alpha,value)






