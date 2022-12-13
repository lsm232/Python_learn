import torch
import torch.nn as nn

class DeformConv2d(nn.Module):
    def __init__(self,inc,outc,kernel_size=3,padding=1,stride=1,bias=None,modulation=False):
        super(DeformConv2d, self).__init__()
        self.kersize=kernel_size
        self.padding=padding
        self.stride=stride
        self.zero_padding=nn.ZeroPad2d(padding)

        self.conv=nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,bias=bias)
        self.p_conv=nn.Conv2d(inc,kernel_size*kernel_size*2,kernel_size=3,stride=1,padding=1)

        nn.init.constant_(self.p_conv.weight,0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation=modulation
        if self.modulation:
            self.m_conv=nn.Conv2d(inc,kernel_size*kernel_size,kernel_size=3,padding=1,stride=stride)
            