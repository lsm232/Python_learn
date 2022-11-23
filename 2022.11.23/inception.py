import torch
import torch.nn as nn

class basic_layer(nn.Module):
    def __init__(self,indims,outdims,kersize):
        super(basic_layer, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(indims,outdims,kersize,padding=kersize//2),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.layer(x)

class inception(nn.Module):
    def __init__(self,in_dims,hid_1_1,hid_2_1,hid_2_3,hid_3_1,hid_3_5,hid_4_1):
        super(inception, self).__init__()
        self.branch1=nn.Sequential(
            basic_layer(in_dims,hid_1_1,1),
        )
        self.branch2=nn.Sequential(
            basic_layer(in_dims,hid_2_1,1),
            basic_layer(hid_2_1,hid_2_3,3)
        )
        self.branch3=nn.Sequential(
            basic_layer(in_dims,hid_3_1,1),
            basic_layer(hid_3_1,hid_3_5,5),
        )
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,padding=1),
            basic_layer(in_dims,hid_4_1,1)
        )
    def forward(self,x):
        b1=self.branch1(x)
        b2=self.branch2(x)
        b3=self.branch3(x)
        b4=self.branch4(x)
        out=torch.cat([b1,b2,b3,b4],dim=1)
        return out
    

