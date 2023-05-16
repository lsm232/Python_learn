import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

#超参
lr_rate=0.01
n_hidden_1=256
n_hidden_2=128
n_input=784
epochs=20
batch_size=32
display_step=5

#构建网络
x=tf.placeholder(tf.float32,[None,n_input])
y=x
weights={'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
         'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
         'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
         'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
         }
bias={'encoder_b1':tf.Variable(tf.zeros([n_hidden_1])),
      'encoder_b2':tf.Variable(tf.zeros([n_hidden_2])),
      'decoder_b1':tf.Variable(tf.zeros([n_hidden_1])),
      'decoder_b2':tf.Variable(tf.zeros([n_input])),
      }
def encoder(x):
    layer1=tf.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),bias['encoder_b1']))
    layer2=tf.sigmoid(tf.add(tf.matmul(layer1,weights['encoder_h2']),bias['encoder_b2']))
    return layer2
def decoder(x):
    layer1 = tf.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), bias['decoder_b1']))
    layer2 = tf.sigmoid(tf.add(tf.matmul(layer1, weights['decoder_h2']), bias['decoder_b2']))
    return layer2
out=encoder(x)
pred=decoder(out)
cost=tf.reduce_mean(tf.pow(y-pred,2))
optimizer=tf.train.RMSPropOptimizer(lr_rate).minimize(cost)

#train
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    for epoch in range(epochs):
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_x})
    if (epoch+1)%display_step==0:
        print('loss=',c)
        order=int(random.randint(0,mnist.test.num_examples))
        rec=sess.run(pred,feed_dict={x:mnist.test.images[order-1:order]})
        f,a=plt.subplots(1,2)
        a[0].imshow(np.reshape(rec,(28,28)))
        a[1].imshow(np.reshape(mnist.test.images[order-1:order],(28,28)))
        plt.show()
    print('ok')
