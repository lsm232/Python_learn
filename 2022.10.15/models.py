import torch
import torch.nn as nn
from .parse_config import *
from typing import List

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








class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416),verbose=False):
        super(Darknet, self).__init__()
        self.input_size=[img_size]*2 if isinstance(img_size,int) else img_size
        self.module_defs=parse_model_cfg(cfg)
        self.module_list,self.routs=create_modules(self.module_defs,img_size)