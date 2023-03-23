import numpy as np
import tensorflow as tf
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
# colors=['r' if l==0 else 'b' for l in Y ]
# plt.scatter(X[:,0],X[:,1],c=colors)
# plt.show()

input_features=tf.placeholder(dtype=tf.float32,shape=[None,2])
input_labels=tf.placeholder(dtype=tf.float32,shape=[None,1])

w=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
pred=tf.matmul(input_features,w)+b
out=tf.nn.sigmoid(pred)
loss=tf.reduce_mean(-(input_labels*tf.log(out)+(1-input_labels)*tf.log(1-out)))
optimizer=tf.train.AdamOptimizer(0.002).minimize(loss)


epochs=100
batchs=10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        err=0.
        for inter in range(int(len(Y)/batchs)):
            x1=X[inter*batchs:(inter+1)*batchs,:]
            y1=Y[inter*batchs:(inter+1)*batchs].reshape(-1,1)
            _,cost=sess.run([optimizer,loss],feed_dict={input_features:x1,input_labels:y1})
            err+=cost/len(Y)
        print('epoch:',err)


    trainx,trainy=generate(100,mean,cov,[3.0],True)
    colors=['g' if l==0 else 'b'   for l in trainy]
    plt.scatter(trainx[:,0],trainx[:,1],c=colors)
    x=np.linspace(-1,8,200)
    y=-x*(sess.run(w)[0]/sess.run(w)[1])-sess.run(b)/sess.run(w)[1]
    plt.plot(x,y)
    plt.show()


