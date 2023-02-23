import tensorflow as tf
import numpy as np

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
    saver.restore(sess,'./model.cpkt')
    print(sess.run(w))