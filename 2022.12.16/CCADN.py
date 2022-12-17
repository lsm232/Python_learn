import torch
import torch.nn as nn


class CBR_block(nn.Module):
    def __init__(self,in_dims,out_dims):
        super(CBR_block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.layer(x)
    
class module(nn.Module):
    def __init__(self,in_dims):
        super(module, self).__init__()
        self.first_layer=CBR_block(in_dims,in_dims)
        self.layer=nn.Sequential(
            CBR_block(in_dims,in_dims),
            CBR_block(in_dims,in_dims),
        )
        self.relu=nn.ReLU(True)
    def forward(self,x):
        res=self.first_layer(x)
        x=self.layer(x)
        x=self.relu(x+res)
        return res,x

class generator_CCADN(nn.Module):
    def __init__(self):
        super(generator_CCADN, self).__init__()

        self.layer0=nn.Sequential(
            CBR_block(1,128),
            CBR_block(128,128),
            CBR_block(128,128),
        )
        self.m1=module(128)
        self.m2=module(128)
        self.m3=module(128)
        self.m4=module(128)
        self.m5=module(128)
        self.m6=module(128)
        self.layer7=nn.Sequential(
            CBR_block(896,128),
            CBR_block(128,128),
        )
        self.layer8=nn.Conv2d(128,1,kernel_size=3,padding=1,stride=1,bias=True)
    def forward(self,x):
        identity=x
        x=self.layer0(x)
        res1,x=self.m1(x)
        res2,x=self.m2(x)
        res3,x=self.m3(x)
        res4,x=self.m4(x)
        res5,x=self.m5(x)
        res6,x=self.m6(x)
        res=torch.cat([res1,res2,res3,res4,res5,res6,x],dim=1)
        x=self.layer7(res)
        x=self.layer8(x)
        out=identity+x
        return out

net=generator_CCADN()
x=torch.rand(size=(2,1,256,256))
out=net(x)
c=1