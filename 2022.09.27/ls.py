import torch
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(5)

x=torch.rand(size=(4,4))
y=np.random.randn(4,4)
z=torch.Tensor(y)
print(x)
print(y)
print(z)