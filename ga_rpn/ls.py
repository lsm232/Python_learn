import torch
import torch.nn as nn
import functools

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class block(nn.Module):
    def __init__(self,in_dims,kersizes=[3,3,1],strides=[1,1,1],dilated_rates=[1,3,1]):
        super(block, self).__init__()
        assert len(kersizes)==len(strides)
        assert len(strides)==len(dilated_rates)
        layer=[]
        for i in range(len(kersizes)):
            layer.append(nn.Conv2d(in_channels=in_dims,out_channels=in_dims,kernel_size=kersizes[i],stride=strides[i],padding=(kersizes[i]+(kersizes[i]-1)*(dilated_rates[i]-1))//2,dilation=dilated_rates[i]))
            if kersizes[i]>1:
                layer.append(nn.InstanceNorm2d(in_dims))
                layer.append(nn.ReLU(True))
        self.layer=nn.Sequential(*layer)
    def forward(self,x):
        res=x
        return res+self.layer(x)

class RCDC_block(nn.Module):
    def __init__(self,in_dims):
        super(RCDC_block, self).__init__()
        self.layer1=block(in_dims,kersizes=[3,1],strides=[1,1],dilated_rates=[3,1])
        self.layer2=block(in_dims,kersizes=[3,3,1],strides=[1,1,1],dilated_rates=[1,3,1])
        self.layer3=block(in_dims,kersizes=[3,3,1],strides=[1,1,1],dilated_rates=[1,3,1])
        self.layer4=block(in_dims,kersizes=[3,3,3,1],strides=[1,1,1,1],dilated_rates=[1,3,5,1])
    def forward(self,x):
        l1=self.layer1(x)
        l2=self.layer2(l1)
        l3=self.layer3(l2)
        l4=self.layer4(l3)
        return l4

class ResnetGenerator_lsm3(nn.Module):
    def __init__(self, is_train, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,padding_type='reflect'):
        super(ResnetGenerator_lsm3, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        self.layer1=nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 32, kernel_size=7, padding=0, bias=False),
            norm_layer(32),
            nn.ReLU(True),
        )
        self.down1=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1,bias=False),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,kernel_size=1,bias=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, bias=True),
        )

        self.RCDC_block=RCDC_block(128)

        self.deconv1=nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.deconv1_back_1x1=nn.Sequential(
            nn.Conv2d(128,64,1,bias=False),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        )
        self.deconv2_back_1x1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
        )

        self.resblock1=ResnetBlock(
            32, padding_type=padding_type, norm_layer=norm_layer,
            use_dropout=use_dropout, use_bias=use_bias
        )

        self.out_conv=nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 1, kernel_size=7, padding=0, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self,x):
        l1=self.layer1(x)
        d1=self.down1(l1)
        d2=self.down2(d1)
        r1=self.RCDC_block(d2)

        d2=self.deconv1(d2)
        r1=self.deconv1(r1)
        d2_r1=torch.cat([d2,r1],dim=1)
        d2_r1=self.deconv1_back_1x1(d2_r1)

        d1=self.deconv2(d1)
        d2_r1=self.deconv2(d2_r1)
        d1_d2_r1=torch.cat([d1,d2_r1],dim=1)
        d1_d2_r1=self.deconv2_back_1x1(d1_d2_r1)
        d1_d2_r1=self.resblock1(d1_d2_r1)

        out=self.out_conv(l1+d1_d2_r1)
        return out

