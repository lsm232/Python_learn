import torch
import torch.nn as nn

class RED_CNN(nn.Module):
    def __init__(self,out_chans):
        super(RED_CNN, self).__init__()
        self.conv1=nn.Conv2d(1,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.conv2=nn.Conv2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.conv3=nn.Conv2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.conv4=nn.Conv2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.conv5=nn.Conv2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)

        self.deconv1=nn.ConvTranspose2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.deconv2=nn.ConvTranspose2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.deconv3=nn.ConvTranspose2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.deconv4=nn.ConvTranspose2d(out_chans,out_chans,kernel_size=5,stride=1,padding=0,bias=True)
        self.deconv5=nn.ConvTranspose2d(out_chans,1,kernel_size=5,stride=1,padding=0,bias=True)

        self.relu=nn.ReLU(True)
    def forward(self,x):
        res1=x
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        res2=x
        x=self.relu(self.conv3(x))
        x=self.relu(self.conv4(x))
        res3=x
        x=self.relu(self.conv5(x))

        x=self.deconv1(x)
        x=self.relu(x+res3)
        x=self.relu(self.deconv2(x))
        x=self.deconv3(x)
        x=self.relu(x+res2)
        x=self.relu(self.deconv4(x))
        x=self.deconv5(x)
        x=self.relu(x+res1)
        return x
