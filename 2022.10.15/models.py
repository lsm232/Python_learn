import torch
import torch.nn as nn
from .parse_config import *
from typing import List
from .layers import *

def create_modules(module_defs:list,img_size):
    img_size=[img_size]*2 if isinstance(img_size,int) else img_size
    module_list=nn.ModuleList()
    out_filters=[3]
    yolo_index=-1
    routs=[]
    module_defs.pop(0)

    for i,mdef in enumerate(module_defs):
        modules=nn.Sequential()
        if mdef["type"]=="convolutional":
            kersize=mdef["size"]
            stride=mdef["stride"]
            filters=mdef["filters"]
            bn=mdef["batch_normalize"]

            modules.add_module("Conv2d",nn.Conv2d(out_filters[-1],filters,kernel_size=kersize,stride=stride,padding=kersize//2 if mdef["pad"] else 0,bias=not bn))

            if bn:
                modules.add_module("Batchnorm2d",nn.BatchNorm2d(filters))
            else:
                routs.append(i)

            if mdef["activation"]=="leaky":
                modules.add_module("activation",nn.LeakyReLU(0.1,True))
            else:
                pass

        elif mdef["type"]=="maxpool":
            kersize=mdef["size"]
            stride=mdef["stride"]
            modules=nn.MaxPool2d(kernel_size=kersize,stride=stride,padding=(kersize-1)//2)

        elif mdef["type"]=="upsample":
            modules=nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"]=="route":
            layers=mdef["layers"]
            filters=sum([out_filters[l if l<0 else l+1]for l in layers])
            routs.extend([i+l if l<0 else l for l in layers])
            modules=FeatureConcat(layers=layers)

        elif mdef["type"]=="shortcut":
            layers=mdef["from"]
            filters=out_filters[-1]
            routs.append(i+layers[0])
            modules=WeightedFeatureFusion(layers=layers)

        elif mdef["type"]=="yolo":
            yolo_index+=1
            stride=[32,16,8]
            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        module_list.append(modules)
        out_filters.append(filters)

    routs_binary=[False]*len(module_defs)
    for i in routs:
        routs_binary[i]=True
    return module_list,routs_binary

class YOLOLayer(nn.Module):
    def __init__(self,anchors,nc,img_size,stride):
        super(YOLOLayer, self).__init__()
        self.anchors=torch.Tensor(anchors)
        self.na=len(anchors)
        self.nc=nc
        self.no=nc+5
        self.stride=stride
        self.nx,self.ny,self.ng=0,0,(0,0)
        self.anchor_vec=self.anchors/self.stride
        self.anchor_wh=self.anchor_vec.view(1,self.na,1,1,2)
        self.grid=None
    def create_grids(self,ng=(13,13),device="cpu"):
        self.nx,self.ny=ng
        self.ng=torch.tensor(ng,dtype=torch.float)

        if not self.training:
            yv,xv=torch.meshgrid([torch.arange(self.ny),torch.arange(self.nx)])
            self.grid=torch.stack((xv,yv),2).view((1,1,self.ny,self.nx,2)).float()

    def forward(self,p):
        bs,_,ny,nx=p.shape
        if(self.nx,self.ny)!=(nx,ny) or self.grid is None:
            self.create_grids((nx,ny))
        p=p.view(bs,self.na,self.no,self.ny,self.nx).permute(0,1,3,4,2).contiguous()
        if self.training:
            return p
        else:
            io=p.clone()
            io[...,:2]=torch.sigmoid(io[...,:2])+self.grid
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[...,:4]*=self.stride
            torch.sigmoid_(io[...,4:])
            return io.view(bs, -1, self.no), p







class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416),verbose=False):
        super(Darknet, self).__init__()
        self.input_size=[img_size]*2 if isinstance(img_size,int) else img_size
        self.module_defs=parse_model_cfg(cfg)
        self.module_list,self.routs=create_modules(self.module_defs,img_size)