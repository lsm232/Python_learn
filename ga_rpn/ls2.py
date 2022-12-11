from ga_rpn.ls import *
import torch

net=ResnetGenerator_lsm3(is_train=True,input_nc=1,output_nc=1)
x=torch.rand(2,1,64,64)
out=net(x)
c=1