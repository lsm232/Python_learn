import numpy as np
int2binary={}
binary_dim=8
largest_number=256
binary=np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)