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
        
