import torch.nn as nn
import torch.nn.functional as F
from typing import List,Tuple,Dict
from torch import Tensor
from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self,backbone,return_layers):
        if not set(return_layers).issubset([layer_name for layer_name,_ in backbone.named_children()]):
            raise ValueError("return layers not in backbone")
        original_return_layers=return_layers
        layers=OrderedDict()

        for name,module in backbone.named_children():
            layers[name]=module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super.__init__(layers)
        self.return_layers=original_return_layers
    def forward(self,x):
        out=OrderedDict()
        for name,module in self.items():
            x=module(x)
            if name in self.return_layers:
                out[name]=x
        return out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self,in_channels_list,out_channels,extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks=nn.ModuleList()
        self.layer_blocks=nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels==0:
                continue
            inner_module=nn.Conv2d(in_channels,out_channels,1)
            layer_module=nn.Conv2d(out_channels,out_channels,3,padding=1)
            self.inner_blocks.append(inner_module)
            self.layer_blocks.append(layer_module)

        for m in self.children():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,a=1)
                nn.init.constant_(m.bias,0)

        self.extra_blocks=extra_blocks

    def get_result_from_inner_blocks(self,x,idx):
        num_blocks=len(self.inner_blocks)
        i=0
        out=x
        if idx<0:
            idx=num_blocks-idx
        for module in self.inner_blocks:
            if i==idx:
                out=module(x)
                return out

            i+=1
        return out

    def get_result_from_layer_blocks(self,x,idx):
        num_blocks=len(self.layer_blocks)
        i=0
        out=x
        if idx<0:
            idx=num_blocks-idx
        for module in self.layer_blocks:
            if i==idx:
                out=module(x)
                return out
            i=i+1







class LastLevelMaxPool(nn.Module):
    def forward(self,featrues:List[Tensor],names:List[str])->Tuple[List[Tensor],List[str]]:
        names.append("pool")
        featrues.append(F.max_pool2d(featrues[-1],1,2,0))
        return featrues,names

class BackboneWithFPN(nn.Module):
    def __init__(self,
                 backbone:nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True,
                 ):
        super(BackboneWithFPN, self).__init__()
        if extra_blocks is None:
            extra_blocks=LastLevelMaxPool()
        if re_getter is True:
            assert return_layers is not None
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else
            self.body=backbone

        self.fpn=FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels=out_channels
    def forward(self,x):
        x=self.body(x)
        x=self.fpn(x)
        return x

