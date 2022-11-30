import torch

def channel_shuffle(x,groups):
    b,c,h,w=x.shape
    channels_per_group=c//groups
    x=x.reshape(b,groups,channels_per_group,h,w)
    x=x.transpose(1,2)
    x=x.reshape(b,c,h,w)
    return x