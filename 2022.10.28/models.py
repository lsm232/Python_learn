import torch
import torch.nn as nn
from layers import *
from parse_config import *
ONNX_EXPORT = False


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

            modules.add_module('conv2d',nn.Conv2d(kernel_size=kersize,stride=stride,in_channels=output_filters[-1],out_channels=filters,padding=kersize//2 if mdef['pad'] else 0,bias=not bn))

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
            routs.extend([m+i if m<0 else m for m in layers])
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
        routs_binary=True
    return module_list,routs_binary


class YOLOLayer(nn.Module):
    def __init__(self,anchors,stride,nc,img_size):
        super(YOLOLayer, self).__init__()
        self.anchors=torch.Tensor(anchors)
        self.stride=stride
        self.na=len(anchors)
        self.nc=nc
        self.no=nc+5 #x,y,w,h,objectness

        self.nx,self.ny,self.ng=0,0,(0,0)  #用来干嘛的
        self.anchor_vec=self.anchors/stride
        self.anchor_wh=self.anchor_vec.view(1,self.na,1,1,2)
        self.grid=None

    def create_grids(self,ng=(13,13),device='cpu'):
        self.nx,self.ny=ng
        x=torch.arange(self.nx)
        y=torch.arange(self.ny)
        grid_x,grid_y=torch.meshgrid([x,y])
        self.grid=torch.stack((grid_y,grid_x),2).view(1,1,self.ny,self.nx,2).float()




    def forward(self,p):
        bs,_,ny,nx=p.shape
        if (self.nx,self.ny)!=(nx,ny) or self.grid is None:
            self.create_grids((nx,ny),p.device)

        p=p.view(bs,self.na,self.no,self.ny,self.nx).permute(0,1,3,4,2).contiguous()

        if self.training:
            return p
        else:
            io=p.clone()
            io[...,:2]=torch.sigmoid(io[...,:2])+self.grid
            io[...,2:4]=torch.exp(io[...,2:4])*self.anchor_wh
            io[...,:4]*=self.stride
            torch.sigmoid_(io[...,4:])
            return io.view(bs,-1,self.no),p


def get_yolo_layers(self):
    return [i  for i,m in enumerate(self.module_list) if m.__class__.__name__=='YOLOLayer']







class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416)):
        super(Darknet, self).__init__()
        self.input_size=[img_size]*2 if isinstance(img_size,int) else img_size
        self.module_defs=parse_model_cfg(cfg)
        self.module_list,self.routs=create_modules(self.module_defs,self.input_size)
        self.yolo_layers=get_yolo_layers(self)
    def forward(self,x):
        yolo_out,out=[],[]
        for i,module in enumerate(self.module_list):
            name=module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:
                x=module(x,out)
            elif name=='YOLOLayer':
                yolo_out.append(x)
            else:
                x=module(x)
            out.append(x if self.routs[i] else [])

            if self.training:
                return yolo_out
            else:
                x,p=zip(*yolo_out)
                x=torch.cat(x,1)
                return x,p








