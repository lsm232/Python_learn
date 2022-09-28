import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import os
from .feature_pyramid_network import *


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,kersize,downsample,norm_layer):
        super(Bottleneck, self).__init__()
        if norm_layer==None:
            norm_layer=nn.BatchNorm2d
        self.conv1=nn.Conv2d(in_channels,out_channels,1,1)
        nn.bn1=norm_layer(out_channels)

        self.conv2=nn.Conv2d(out_channels,out_channels,kersize,1,1)
        nn.bn2=norm_layer(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1, 1)
        nn.bn3 = norm_layer(out_channels)

        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out+=identity
        out=self.relu(out)  #这里的写法比较奇怪，先残差，再relu
        return out

class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=False,norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is not None:
            norm_layer=nn.BatchNorm2d
        self._norm_layer=norm_layer
        self.include_top=include_top
        self.in_channels=64

        self.conv=nn.Sequential(
            nn.Conv2d(3,self.in_channels,7,2,3),
            norm_layer(self.in_channels),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        )
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self,block,channel,block_num,stride=1):
        norm_layer=self._norm_layer
        downsample=None
        if stride!=1 or self.in_channels!=block.expansion*channel:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,channel*block.expansion,1,stride=stride),
                norm_layer(channel*block.expansion),
            )
        layers=[]
        layers.append(block(self.in_channels,channel,downsample,stride,norm_layer))
        self.in_channels=channel*block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x=self.avgpool(x)
            x=torch.flatten(x,1)
            x=self.fc(x)
        return x

def overwrite_eps(model,eps):
    for module in model.modules():
        if isinstance(module,FrozenBatchNorm2d):
            module.eps=eps

def resnet50_fpn_backbone(pretrain_path='',norm_layer=None,layer_to_train=5,returned_layers=None,extra_blocks=None):
    resnet_backbone=ResNet(block=Bottleneck,blocks_num=[3,4,6,3],include_top=False,norm_layer=FrozenBatchNorm2d)
    if isinstance(norm_layer,FrozenBatchNorm2d):
        overwrite_eps(norm_layer,0.0)
    if pretrain_path=='':
        assert os.path.exists(pretrain_path),"{} is not exist".format(pretrain_path)
        resnet_backbone.load_state_dict(torch.load(pretrain_path))
    assert 0<=layer_to_train<=5
    layer_to_train_list=['layer4','layer3', 'layer2', 'layer1', 'conv'][:layer_to_train]

    for name,parameter in resnet_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layer_to_train_list]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks=LastLevelMaxPool()

    if returned_layers is None:
        returned_layers=[1,2,3,4]

    assert min(returned_layers) > 0 and max(returned_layers) < 5
    returned_layers={f'layer{k}':str(v) for v,k in enumerate(returned_layers)}

    in_channels_stage2=resnet_backbone.in_channels//Bottleneck.expansion//2
    in_channels_list=[in_channels_stage2*2**(i-1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet_backbone,returned_layers,in_channels_list,out_channels,extra_blocks)











