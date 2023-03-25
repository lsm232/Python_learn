import numpy as np
import tensorflow as tf

def generate(sample_size,mean,cov,diff,regression,num_class):
    sampls_per_class=int(sample_size/num_class)
    x0=np.random.multivariate_normal(mean,cov,sampls_per_class)
    y0=np.zeros(sampls_per_class)

    for ci,d in enumerate(diff):
        x1=np.random.multivariate_normal(mean+d,cov,sampls_per_class)
        y1=ci+1+y0

        x=np.concatenate([x0,x1])
        y=np.concatenate([y0,y1])

    if regression==False:
        class_ind=[y0==class_number for class_number in range(num_class)]
        y=np.asarray(np.hstack([class_ind]),dtype=np.float32)
    return x,y

