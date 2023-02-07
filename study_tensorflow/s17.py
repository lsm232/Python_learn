import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate(samples,num_class,mean,cov,diff,regression):
    samples_per_class=int(samples)/num_class
    x0=np.random.multivariate_normal(mean,cov,samples_per_class)
    y0=np.zeros([samples_per_class])

    for i,dif in enumerate(diff):
        x1=np.random.multivariate_normal(mean+dif,cov,samples_per_class)
        y1=(i+1)+np.zeros(samples_per_class)

        x0=np.concatenate([x0,x1])
        y0=np.concatenate([y0,y1])

    if regression==False:
        class_id=[y0==class_number for class_number in range(num_class)]
        y0=np.hstack([class_id],dtype=np.float32)

    return x0,y0

