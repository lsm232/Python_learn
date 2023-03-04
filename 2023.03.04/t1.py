import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride,downsaple):

        super(Bottleneck, self).__init__()
        self.downsample=downsaple
        self.layer=nn.Sequential(
            nn.Conv2d(in_planes,planes,1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes,3,stride,1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes*Bottleneck.expansion, 3, stride, 1),
            nn.BatchNorm2d(planes*Bottleneck.expansion),
        )
    def forward(self,x):
        identity=x
        if self.downsample is not  None:
            identity=self.downsample(x)
        return F.relu(identity+self.layer(x))

class FPN(nn.Module):
    def __init__(self,layers=[3,4,6,3]):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1=nn.Conv2d(3,64,7,2,3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu1=nn.ReLU(True)
        self.maxpool=nn.MaxPool2d(3,2,1)

        self.layer1=self._make_layer(64,layers[0],1)
        self.layer2=self._make_layer(128,layers[1],2)
        self.layer3=self._make_layer(256,layers[2],2)
        self.layer4=self._make_layer(512,layers[3],2)

        self.toplayer=nn.Conv2d(512*4,256,1)

        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)

        self.latlayer1=nn.Conv2d(1024,256,1)
        self.latlayer2=nn.Conv2d(512,256,1)
        self.latlayer3=nn.Conv2d(256,256,1)

    def _make_layer(self,planes,num_blocks,stride):

        downsample=None
        if stride!=1 or self.in_planes!=Bottleneck.expansion*planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_planes,planes*Bottleneck.expansion,1,stride),
                nn.BatchNorm2d(planes*Bottleneck.expansion),
            )
        layers=[Bottleneck(self.in_planes,planes,stride,downsample)]
        self.in_planes=Bottleneck.expansion*planes
        for i in range(1,num_blocks):
            layers.append(Bottleneck(self.in_planes,planes,1,None))

        return nn.Sequential(*layers)

    def _upsample_add(self,x,y):
        b,c,h,w=y.shape
        return y+F.upsample(x,size=(h,w),mode='bilinear')

    def forward(self,x):
        c1=self.maxpool(self.relu1(self.bn1(self.conv1(x))))
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)

        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer3(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer1(c2))

        p4=self.smooth3(p4)
        p3=self.smooth2(p3)
        p2=self.smooth1(p2)

        return p2,p3,p4,p5

net=FPN()
c=1
