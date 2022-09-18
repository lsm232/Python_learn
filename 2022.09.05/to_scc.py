import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(trainA_path,trainB_path):
    filesA = sorted(os.listdir(trainA_path))
    filesB = sorted(os.listdir(trainB_path))
    assert len(filesA) == len(filesB)
    filesA_path=[]
    filesB_path=[]

    for fileA,fileB in zip(filesA,filesB):
        filesA_path.append(trainA_path+'/'+fileA)
        filesB_path.append(trainB_path+'/'+fileB)

    imgA=Image.open(filesA_path[0]).convert('F')
    imgB=Image.open(filesB_path[0]).convert('F')
    # plt.subplot(1,2,1)
    # plt.imshow(np.asarray(imgA),cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.asarray(imgB), cmap='gray')
    # plt.show()

    nums = len(filesA)
    stack_imgAs = torch.rand((nums, 1,torch.tensor(np.asarray(imgA)).shape[-2:][0],torch.tensor(np.asarray(imgA)).shape[-2:][1]),dtype=torch.float32)
    stack_imgBs = torch.rand((nums, 1,torch.tensor(np.asarray(imgA)).shape[-2:][0],torch.tensor(np.asarray(imgA)).shape[-2:][1]),dtype=torch.float32)

    i=0
    for imgA_path,imgB_path in zip(filesA_path,filesB_path):
        imgA=torch.tensor(np.asarray(Image.open(imgA_path).convert('F'))).unsqueeze(0)
        imgB=torch.tensor(np.asarray(Image.open(imgB_path).convert('F'))).unsqueeze(0)
        stack_imgAs[i]=imgA
        stack_imgBs[i]=imgB
        i+=1

    return stack_imgAs,stack_imgBs


#example
a,b=load_data(r'J:\自监督所用数据\ct_2022.06.15\stage1\test\low',r'J:\自监督所用数据\ct_2022.06.15\stage1\test\high')










