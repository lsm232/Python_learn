import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_dims,inter_dims,stride,downsample):
        super(Bottleneck, self).__init__()
        self.stride=stride
        self.downsample=downsample

        self.layer=nn.Sequential(
            nn.Conv2d(in_dims,inter_dims,kernel_size=1),
            nn.BatchNorm2d(inter_dims),
            nn.ReLU(True),
            nn.Conv2d(inter_dims,inter_dims,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(inter_dims),
            nn.ReLU(True),
            nn.Conv2d(inter_dims,inter_dims*self.expansion,kernel_size=1),
            nn.BatchNorm2d(in_dims*self.expansion),
        )

        self.relu=nn.ReLU(True)

    def forward(self,x):
        identity=x
        x=self.layer(x)
        if self.downsample is not None:
            identity=self.downsample(identity)
        out=x+identity
        return self.relu(out)

class FPN(nn.Module):
    def __init__(self,layers):
        super(FPN, self).__init__()
        self.inplanes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(64,layers[0])
        self.layer2=self._make_layer(128,layers[0],2)
        self.layer3=self._make_layer(256,layers[0],2)
        self.layer4=self._make_layer(512,layers[0],2)

        self.toplayer=nn.Conv2d(2048,256,1,1,0)
        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)

        self.latlayer1=nn.Conv2d(1024,256,1)
        self.latlayer2=nn.Conv2d(512,256,1)
        self.latlayer3=nn.Conv2d(256,256,1)

    def _make_layer(self,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*Bottleneck.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,planes*Bottleneck.expansion,1,stride=stride),
                nn.BatchNorm2d(planes*Bottleneck.expansion)
            )
        layers=[]
        layers.append(Bottleneck(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*Bottleneck.expansion
        for i in range(1,blocks):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)

    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.upsample(x,(H,W))+y

    def forward(self,x):
        c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)

        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))

        p4=self.smooth1(p4)
        p3=self.smooth1(p3)
        p2=self.smooth1(p2)
        return p2,p3,p4,p5




