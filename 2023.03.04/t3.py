import torch

x=torch.randn([1,1,4,4])
y=x.roll(shifts=[1,1],dims=[2,3])
z=y.roll(shifts=[1,1],dims=[2,3])

c=1