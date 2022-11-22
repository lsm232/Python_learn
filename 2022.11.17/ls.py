import torch

a=torch.Tensor([1,2,3])
b=torch.Tensor([4,5,6])
c,d=a.chunk(2,dim=0)
m=d
d+=1
m=1





