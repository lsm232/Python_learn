import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pylab

mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)
print('shape=',mnist.train.images.shape)
im=mnist.train.images.reshape(-1,28,28)
# pylab.imshow(im[1])
# pylab.show()


lr_rate=0.001
training_epochs=25
batch_size=10
display_step=10

n_hidden_1=256
n_hidden_2=256
n_input=784
n_classes=10

x=tf.placeholder(np.float32,[None,n_input])
y=tf.placeholder(np.float32,[None,n_classes])

weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'h3':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
bias={
    'b1':tf.Variable(tf.zeros([n_hidden_1])),
    'b2':tf.Variable(tf.zeros([n_hidden_2])),
    'b3':tf.Variable(tf.zeros([n_classes]))
}


def mlp(x,weights,bias):
    layer_1=tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),bias['b1']))
    layer_2=tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']),bias['b2']))
    layer_3=tf.matmul(layer_2,weights['h3'])+bias['b3']
    return layer_3

pred=mlp(x,weights,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(lr_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost=0.

        for i in range(500):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost=c/batch_size

            if (epoch+1)%display_step==0:
                print('epoch:',"%4d"%(epoch+1),'cost={:.4f}'.format(avg_cost))
