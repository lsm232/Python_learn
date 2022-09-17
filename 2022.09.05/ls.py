import torch


a=torch.tensor([[1,2,3],[4,5,6]])
idx=torch.tensor([[0],[1]])
idx2=torch.tensor([[1],[2]])
b=a[idx,idx2]
c=1