import numpy as np
import matplotlib.pyplot as plt


mean=np.random.randn(2)
cov=np.eye(2)
samples_per_class=500
dif=4
x0=np.random.multivariate_normal(mean,cov,samples_per_class)
y0=np.zeros(samples_per_class)

x1=np.random.multivariate_normal(mean+dif,cov,samples_per_class)
y1=np.ones(samples_per_class)

x=np.concatenate([x0,x1])
y=np.concatenate([y0,y1])

colors=['r' if l==0 else 'b' for l in y]

plt.scatter(x[:,0],x[:,1],c=colors)
plt.show()