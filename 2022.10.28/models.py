import torch
import torch.nn as nn
from .parse_config import *


class Darknet(nn.Module):
    def __init__(self,cfg,img_size=(416,416)):
        super(Darknet, self).__init__()
        self.input_size=[img_size]*2 if isinstance(img_size,int) else img_size
        self.module_defs=parse_model_cfg(cfg)
        self.module_list,self.routs=create_modules(self.module_defs,self.input_size)
