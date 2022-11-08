import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

import torch.distributed as dist

print(dist.get_rank())