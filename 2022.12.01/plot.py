import matplotlib.pyplot as plt
import numpy as np
import math


n_epochs=200
keep_rate_epochs=100

def rule(z):
    print("z==========:",z)
    return 1.0 - max(0, z - keep_rate_epochs) / float(n_epochs - keep_rate_epochs + 1)

def rule2(z):
    print("z==========:", z)
    return ((1 + math.cos(z * math.pi / n_epochs)) / 2) * (1 - 0.01) + 0.01

x=np.arange(0,200)
y=np.array(list(map(rule2,x)))

plt.plot(x,y)
plt.show()
c=1
