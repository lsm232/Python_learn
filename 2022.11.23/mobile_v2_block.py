import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self,indims,hidden_dims,outdims):
        super(InvertedResidual, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(indims,hidden_dims,kernel_size=1),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU6(True),

            nn.Conv2d(hidden_dims,hidden_dims,kernel_size=3,stride=1,padding=1,groups=hidden_dims),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU6(True),

            nn.Conv2d(hidden_dims,outdims,kernel_size=1),
            nn.BatchNorm2d(outdims),

        )
    def forward(self,x):
        return x+self.layer(x)