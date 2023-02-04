import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
num_classes=2
mean=np.random.randn(num_classes)
cov=np.eye(num_classes)

def generate(sample_size,mean,cov,diff,regression):
    num_classes=2
    samples_per_class=int(sample_size/num_classes)

    x0=np.random.multivariate_normal(mean,cov,samples_per_class)
    y0=np.zeros(samples_per_class)

    for i,d in enumerate(diff):
        x1=np.random.multivariate_normal(mean+d,cov,samples_per_class)
        y1=np.ones(samples_per_class)

        x0=np.concatenate([x0,x1])
        y0=np.concatenate([y0,y1])

    if regression==False:
        class_ind=[y0==class_number for class_number in range(num_classes)]
        y=np.asarray(np.hstack(class_ind),dtype=np.float32)
    return x0,y0



X,Y=generate(1000,mean,cov,[3.0],False)
colors=['r' if l ==0 else 'b'  for l in Y[:]]
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()