import torch
import torch.nn as nn

def add_tcb(cfg):
    feature_up_layers=[]
    feature_bottom_layers=[]
    feature_de_layers=[]
    for i,k in enumerate(cfg):
        feature_up_layers.append([nn.Conv2d(cfg[i],256,kernel_size=3,padding=1,stride=1,bias=False),nn.ReLU(True),nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)])
        feature_bottom_layers.append([nn.ReLU(True),nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(True)])
        if i!=0:
            feature_de_layers.append(nn.ConvTranspose2d(256,256,4,2))
    return feature_up_layers,feature_bottom_layers,feature_de_layers