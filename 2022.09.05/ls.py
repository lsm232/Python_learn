import torch


a=torch.tensor([[1,2],[4,5]])
b=torch.tensor([[0,2],[0,5]])
c=torch.stack([a,b],dim=2)
b=a.reshape(2,4)
c=1