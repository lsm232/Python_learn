import tensorflow as tf
import numpy as np
import pylab
import matplotlib.pyplot as plt

# saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.save(sess,'./model.pth')
#     saver.restore(sess,'./model.pth')

#data
x=np.linspace(-1,1,100)
y=20*x+0.3*np.random.random(*x.shape)
# plt.plot(x,y,'ro',label='data')
# plt.legend()
# plt.show()
#model front
w=tf.Variable(tf.ones(1))
b=tf.Variable(tf.zeros(1))
d1=tf.placeholder(dtype=np.float32)
target=tf.placeholder(dtype=np.float32)
pred=tf.multiply(w,d1)+b
#model back
cost=tf.reduce_mean(tf.square(pred-target))
lr_rate=0.02
optimizer=tf.train.GradientDescentOptimizer(lr_rate).minimize(cost)
#model train
epochs=20
dislpay=2
init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    plotdata={'batch_size':[],"loss":[]}
    for epoch in range(epochs):
        for l1,l2 in zip(x,y):
            sess.run(optimizer,feed_dict={d1:l1,target:l2})
            los=sess.run(cost,feed_dict={d1:l1,target:l2})
            plotdata['loss'].append(los)
            plotdata['batch_size'].append(epoch)

        if (epoch+1)%dislpay==0:
            print('w:',sess.run(w))
            print('b:',sess.run(b))
    saver.save(sess,'./model.cpkt')
    plt.plot(x,sess.run(w)*x+sess.run(b))
    plt.plot(plotdata['batch_size'],plotdata['loss'])

    plt.show()

c=1