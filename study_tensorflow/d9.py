import torch
import torch.nn as nn
import torch.nn.functional as f

class Bottleneck(nn.Module):
    def __init__(self,in_dims,out_dims,grow_rate):
        super(Bottleneck, self).__init__()
        inter_dims=grow_rate*4
        self.bn1 = nn.BatchNorm2d(in_dims)
        self.conv1=nn.Conv2d(in_dims,inter_dims,1)
        self.bn2 = nn.BatchNorm2d(inter_dims)
        self.conv2=nn.Conv2d(inter_dims,grow_rate,3,1,1)

    def forward(self,x):
        out1=self.conv1(f.relu(self.bn1(x)))
        out2=self.conv2(f.relu(self.bn2(x)))
        out=torch.cat([out2,x],dim=1)
        return out


class Denseblock(nn.Module):
    def __init__(self,in_dims,grow_rate,n_blocks):
        super(Denseblock, self).__init__()
        layers=[]
        for i in range(n_blocks):
            layers+=[Bottleneck(in_dims,0,grow_rate)]
            in_dims+=grow_rate
        self.layer=nn.Sequential(*layers)
    def forward(self,x):
        return self.layer(x)
