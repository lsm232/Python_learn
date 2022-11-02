import torch
import torch.nn as nn
from .parse_config import *



def create_modules(module_defs,img_size):
    module_list=nn.ModuleList()
    output_filters=[3]
    yolo_index=-1
    routs=[]
    module_defs.pop(0)

    for i,mdef in enumerate(module_defs):
        modules=nn.Sequential()
        if mdef["type"]=="convolutional":
            kersize=mdef['size']
            stride=mdef['stride']
            filters=mdef['filters']
            bn=mdef['batch_normalize']

            modules.add_module(nn.Conv2d(kernel_size=kersize,stride=stride,in_channels=output_filters[-1],out_channels=filters,padding=kersize//2 if mdef['pad'] else 0,bias=not bn))

            if bn:
                modules.add_module('bn',nn.BatchNorm2d(filters))
            else:
                routs.append(i)

            if mdef['activation']=='leaky':
                modules.add_module('act',nn.LeakyReLU(0.2,True))
            else:
                pass

        elif mdef['type']=="BatchNorm2d":
            pass

        elif mdef['type']=="maxpool":
            kersize=mdef['size']
            stride=mdef['stride']
            modules.add_module('maxpool',nn.MaxPool2d(kernel_size=kersize,stride=stride))

        elif mdef['type']=='upsample':
            stride=mdef['stride']
            modules.add_module('up',nn.Upsample(scale_factor=stride))

        elif mdef['type']=='route':
            layers=mdef['layers']
            filters=sum([output_filters[m]  if m<0 else m+1 for m in layers ])
            routs.extend([m+i if m<0 else for m in layers])
            modules=FeatureConcat(layers=layers)

        elif mdef['type']=='shortcut':
            layers=mdef['from']
            filters=output_filters[-1]
            routs.append(i+layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef['type']=="yolo":
            yolo_index+=1
            strides=[32,16,8]
            modules=YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=strides[yolo_index])

        module_list.append(modules)
        output_filters.append(filters)

    routs_binary=[False] *len(module_defs)
    for i in routs:
        routs[i]=True
    return module_list,routs_binary






class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416)):
        super(Darknet, self).__init__()
        self.input_size=[img_size]*2 if isinstance(img_size,int) else img_size
        self.module_defs=parse_model_cfg(cfg)
        self.module_list,self.routs=create_modules(self.module_defs,self.input_size)
