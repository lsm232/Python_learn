import torch

px,py=torch.meshgrid(
    torch.arange(-1,2),
    torch.arange(-1,2),
)
p_n=torch.cat([torch.flatten(px),torch.flatten(py)],0)
p_n=p_n.view(1,18,1,1)
c=1