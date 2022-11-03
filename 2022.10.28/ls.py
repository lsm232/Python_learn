import numpy as np
import torch

x=torch.tensor([1,2,3])
y=torch.tensor([4,5,6])
grid_x,grid_y=torch.meshgrid(x,y)
c=1
