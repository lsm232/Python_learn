import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
# x=np.random.random([2])
# y=np.zeros([2])
# z=np.concatenate([x,y])
# z2=np.hstack([x,y])
# c=1


def generate(sample_size,mean,cov,diff,regression):
    num_class=2
    samples_per_class=int(sample_size/num_class)
    X0=np.random.multivariate_normal(mean,cov,samples_per_class)
    Y0=np.zeros(samples_per_class)

    for ci,d in enumerate(diff):
        X1=np.random.multivariate_normal(mean+d,cov,samples_per_class)
        Y1=ci+1+np.zeros(samples_per_class)
        X0=np.concatenate([X0,X1])
        Y0=np.concatenate([Y0,Y1])
    if regression==False:
        class_id=[Y0==class_id for class_id in range(num_class)]
        Y=np.asarray(np.hstack(class_id),dtype=np.float32)

    return X0,Y0

num_dims=2
mean=np.random.randn(num_dims)
cov=np.eye(num_dims)
X,Y=generate(1000,mean,cov,[3.0],True)
c=1
colors=['r' if l==0 else 'b' for l in Y ]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()