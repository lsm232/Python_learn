import torch
import torch.nn as nn
import math

#通道注意机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # tensor_1.expand_as(tensor_2) ：把tensor_1扩展成和tensor_2一样的形状



class cbam_c(nn.Module):
    def __init__(self,in_dims,ratio=16):
        super(cbam_c, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp=nn.Sequential(nn.Conv2d(in_dims,in_dims//ratio,kernel_size=1),nn.ReLU(),nn.Conv2d(in_dims//ratio,in_dims,kernel_size=1))
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class cbam_s(nn.Module):
    def __init__(self,kersize=7):
        super(cbam_s, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=kersize,stride=1,padding=kersize//2)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.layer(out)
        return self.sigmoid(out)
class cbam(nn.Module):
    def __init__(self,in_dims,ratio=16,kersize=7,str='parallel',use=''):  #输入的通道数，mlp的ratio，空间注意力的卷积核尺寸
        super(cbam, self).__init__()
        self.cbam_c=cbam_c(in_dims=in_dims,ratio=ratio)
        self.cbam_s=cbam_s(kersize=kersize)
        self.str=str
        self.use =use

    def forward(self,x):
        if self.str!='parallel': #sequential
            inter=self.cbam_c(x)*x+x
            out=self.cbam_s(inter)*inter+inter
            return out
        else:
            if self.use=="both":
                cx=self.cbam_c(x)*x
                sx=self.cbam_s(x)*x
                out=cx+sx
            elif self.use=="spatial":
                cx = 0
                sx = self.cbam_s(x) * x
                out = cx + sx
            elif self.use=="channel":
                cx = self.cbam_c(x) * x
                sx = 0
                out = cx + sx
            else:
                raise ValueError("choose true style!!!")
            return out


class cbam_ce(nn.Module):
    def __init__(self, in_dims, ratio=16):
        super(cbam_ce, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(in_dims, in_dims // ratio, kernel_size=1), nn.ReLU(),nn.Conv2d(in_dims // ratio, in_dims, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

        kersize_1d=int(abs((math.log(in_dims,2)+1)/2))
        kersize_1d=kersize_1d if kersize_1d%2 else kersize_1d+1
        self.conv1d=nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kersize_1d,padding=int(kersize_1d/2))

        self.a=nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out #b,c,1,1
        out = self.sigmoid(out)

        out2=out.squeeze(-1)  #b,c,1
        out2=out2.permute(0,2,1) #b,1,c
        #
        out2=self.conv1d(out2)
        out2=out2.permute(0,2,1).unsqueeze(-1)
        out2=self.sigmoid(out2)
        return out+out2*self.a


class cbam_se2(nn.Module):
    def __init__(self, kersize=[3,5,7], in_dims=None):
        super(cbam_se2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kersize[0], stride=1, padding=kersize[0] // 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kersize[1], stride=1, padding=kersize[1] // 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kersize[2], stride=1, padding=kersize[2] // 2)
        )
        self.sigmoid = nn.Sigmoid()

        #
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_dims, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.layer1(out)
        out2 = self.layer2(out)
        out3 = self.layer3(out)
        out = self.sigmoid(out1+out2+out3)

        #
        attn = self.conv1_1(x)
        output = out + attn

        return output

class cbam_se(nn.Module):
    def __init__(self,kersize=7,in_dims=None):
        super(cbam_se, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=kersize,stride=1,padding=kersize//2)
        )
        self.sigmoid = nn.Sigmoid()

        #
        self.conv1_1=nn.Sequential(
            nn.Conv2d(in_dims,1,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.layer(out)
        out=self.sigmoid(out)

        #
        attn=self.conv1_1(x)
        output=out+attn

        return output
class cbam_e(nn.Module):
    def __init__(self,in_dims,ratio=16,kersize=7,str='parallel'):  #输入的通道数，mlp的ratio，空间注意力的卷积核尺寸
        super(cbam_e, self).__init__()
        self.cbam_ce=cbam_ce(in_dims=in_dims,ratio=ratio)
        self.cbam_se=cbam_se(kersize=kersize,in_dims=in_dims)
        self.str=str


    def forward(self,x):
        if self.str!='parallel': #sequential
            inter=self.cbam_ce(x)*x+x
            out=self.cbam_se(inter)*inter+inter
            return out

        else:
            cx=self.cbam_ce(x)*x
            sx=self.cbam_se(x)*x
            out=cx+sx
            return out
class cbam_e2(nn.Module):
    def __init__(self,in_dims,ratio=16,kersize=[3,5,7],str='parallel'):  #输入的通道数，mlp的ratio，空间注意力的卷积核尺寸
        super(cbam_e2, self).__init__()
        self.cbam_ce=cbam_ce(in_dims=in_dims,ratio=ratio)
        self.cbam_se=cbam_se2(kersize=kersize,in_dims=in_dims)
        self.str=str


    def forward(self,x):
        if self.str!='parallel': #sequential
            inter=self.cbam_ce(x)*x+x
            out=self.cbam_se(inter)*inter+inter
            return out

        else:
            cx=self.cbam_ce(x)*x
            sx=self.cbam_se(x)*x
            out=cx+sx
            return out
