import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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


input_dim=2
lab_dim=1
input_features=tf.placeholder(dtype=tf.float32,shape=[None,input_dim])
input_labels=tf.placeholder(dtype=tf.float32,shape=[None,lab_dim])
w=tf.Variable(tf.random_normal([input_dim,lab_dim]),name='weight')
b=tf.Variable(tf.zeros([lab_dim]),name='bias')

output=tf.nn.sigmoid(tf.matmul(input_features,w)+b)
cross_entropy=-(input_labels*tf.log(output)+(1-input_labels)*tf.log(1-output))
ser=tf.square(input_labels-output)
loss=tf.reduce_mean(cross_entropy)
err=tf.reduce_mean(ser)
optimizer=tf.train.AdamOptimizer(0.04)
train=optimizer.minimize(loss)

maxepochs=10
batch=25

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxepochs):
        sumerr=0
        for i in range(int(len(y)/batch)):
            x1=x[i*batch:(i+1)*batch,:]
            y1=np.reshape(y[i*batch:(i+1)*batch],[-1,1])
            tf.reshape(y1,[-1,1])

            _,lossval,outputval,errval=sess.run([train,loss,output,err],feed_dict={input_features:x1,input_labels:y1})
            sumerr+=errval

        print('epoch:','%04d'%(epoch+1),'cost=','{:.4f}'.format(lossval),'err=',(sumerr/len(y)/batch))

    x=np.linspace(-1,8,200)
    y=-x*(sess.run(w)[0]/sess.run(w)[1])-sess.run(b)/sess.run(w)[1]
    plt.plot(x,y)
    plt.show()