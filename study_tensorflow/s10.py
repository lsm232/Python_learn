import numpy as np

def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))
    return kernel

ker=make_kernel(2)
print(ker)