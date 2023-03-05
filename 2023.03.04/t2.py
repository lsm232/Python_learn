import torch
import torch.nn as nn

class DetBottleneck(nn.Module):
    def __init__(self,in_planes,planes,extra):
        super(DetBottleneck, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_planes,planes,1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, 3,1,1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes,planes,1),
            nn.BatchNorm2d(planes)
        )
        if extra:
            self.extra_layer=nn.Conv2d(in_planes,planes,1)
    def forward(self,x):
        identity=x
        if self.extra_layer:
            identity=self.extra_layer(x)
        out=self.layer(x)
        return nn.functional.relu(out+identity)