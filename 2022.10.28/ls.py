import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

with open(r'./data/my_train_data.shapes','r') as fid:
    m=fid.read().splitlines()
    for ms in m:
        c=ms.split()
        z=1
