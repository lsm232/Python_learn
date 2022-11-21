import torch

a=torch.arange(0,6).reshape(1,2,3)
b=torch.arange(6,12).reshape(1,2,3)
c=torch.stack([a,b],dim=0)
d=torch.stack([a,b],dim=1)
e=torch.stack([a,b],dim=2)
f=torch.stack([a,b],dim=3)
m=1




