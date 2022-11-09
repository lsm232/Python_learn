import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_chans,out_chans,stride,downsample):
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(in_chans,out_chans,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(out_chans)

        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)

        self.conv3 = nn.Conv2d(out_chans, out_chans*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_chans)

        self.relu=nn.ReLU(inplace=True)

        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out+=identity
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True):
        super(ResNet, self).__init__()
        self.in_channels=64
        self.include_top=include_top

        self.conv1=nn.Conv2d(in_channels=3,out_channels=self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_channels)
        self.relu=nn.ReLU(True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(block,64,blocks_num[0],stride=1)
        self.layer2=self._make_layer(block,128,blocks_num[1],stride=2)
        self.layer3=self._make_layer(block,256,blocks_num[2],stride=2)
        self.layer4=self._make_layer(block,512,blocks_num[3],stride=2)

        if self.include_top:
            self.avgpool=nn.AdaptiveMaxPool2d((1,1))
            self.fc=nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')

    def _make_layer(self,block,channels,block_nums,stride):
        downsample=None
        if stride!=1 or self.channels!=block.expansion*channels:
            downsample=nn.Sequential(
                nn.Conv2d(channels,channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channels*block.expansion),
            )
        layers=[]
        layers.append(block(self.in_channels,channels,stride,downsample))
        self.in_channels=channels*block.expansion

        for _ in range(1, block_nums):
            layers.append(block(self.in_channel, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)