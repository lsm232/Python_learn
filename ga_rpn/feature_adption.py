import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d,MaskedConv2d
from mmcv.runner import BaseModule

class FeatureAdaption(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.1,
                     override=dict(type='Normal', name='conv_adaption', std=0.01))):
        super(FeatureAdaption, self).__init__(init_cfg)
        offset_channels=kernel_size*kernel_size*2
        self.conv_offset=nn.Conv2d(2,deform_groups*offset_channels,1,bias=False)
        self.conv_adaption=DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2,
            deform_groups=deform_groups,
        )
        self.relu=nn.ReLU(True)
    def forward(self,x,shape):
        offset=self.conv_offset(shape.detach())
        x=self.relu(self.conv_adaption(x,offset))
        return x

