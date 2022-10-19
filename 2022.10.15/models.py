import torch
import torch.nn as nn
from .parse_config import *
from typing import List

def create_modules(module_defs:list,img_size):
    img_size=[img_size]*2 if isinstance(img_size,int) else img_size
    module_defs.pop(0)
    output_filters=[3]
    module_list=nn.ModuleList()
    routs=[]
    yolo_index=-1  #???





class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416),verbose=False):
        super(Darknet, self).__init__()
        self.input_size=[img_size]*2 if isinstance(img_size,int) else img_size
        self.module_defs=parse_model_cfg(cfg)
        self.module_list,self.routs=create_modules(self.module_defs,img_size)