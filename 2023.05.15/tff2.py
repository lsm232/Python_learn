import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

train_X=mnist.train.images
train_Y=mnist.train.labels
test_X=mnist.test.images
test_Y=mnist.test.labels

#参数
n_input=784
n_hidden_1=256
n_hidden_2=128
n_classes=10
x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_input])
dropout_keep_prob=tf.placeholder('float')
l2x=tf.placeholder('float',[None,n_hidden_1])
l2y=tf.placeholder('float',[None,n_hidden_1])
l3x=tf.placeholder('float',[None,n_hidden_2])
l3y=tf.placeholder('float',[None,n_classes])
weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'l1_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_1])),
    'l1_out':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
    'l2_h1':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'l2_h2':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_2])),
    'l2_out':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes])),
}
l1_out=tf.nn.sigmoid(tf.matmul(x,weights['h1']))
def noise_l1_autodecoder(layer_1,_weights,_biases,_keep_prob):
    layer_1out=tf.nn.dropout(layer_1,_keep_prob)
    layer_2=tf.sigmoid(tf.matmul(layer_1out,_weights['l1_h2']))
    layer2_out=tf.nn.dropout(layer_2,_keep_prob)
    return tf.sigmoid(tf.matmul(layer2_out,_weights['l1_out']))

l1_reconstruction=noise_l1_autodecoder(l1_out,weights,0,dropout_keep_prob)
l1_cost=tf.reduce_mean(tf.pow(l1_reconstruction-y,2))
l1_optm=tf.train.AdamOptimizer(learning_rate=0.01).minimize(l1_cost)

def l2_autodecoder(layer1_2,_weights,_biases):
    layer1_2out=tf.sigmoid(tf.matmul(layer1_2,_weights['l2_h2']))
    return tf.sigmoid(tf.matmul(layer1_2out,_weights['l2_out']))

l2_out=tf.sigmoid(tf.matmul(l2x,weights['l2_h1']))
l2_reconstruction=l2_autodecoder(l2_out,weights,0)
l2_cost=tf.reduce_mean(tf.pow(l2_reconstruction-l2y,2))
optm2=tf.train.AdamOptimizer(0.01).minimize(l2_cost)

l3_out=tf.matmul(l3x,weights['out'])
l3_cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l3_out,labels=l3y))
l3_optm=tf.train.AdamOptimizer(0.01).minimize(l3_cost)

l1_l2out=tf.nn.sigmoid(tf.matmul(l1_out,weights['l2_h1']))
pred=tf.matmul(l1_l2out,weights['out'])
cost3=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=l3y))
optm3=tf.train.AdamOptimizer(0.001).minimize(cost3)


epochs=50
batch_size=32
display=10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        num_batch=int(mnist.train.num_examples/batch_size)
        total_cost=0.
        for i in range(num_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            batch_xs_noisy=batch_xs+0.3*np.random.randn(batch_size,784)
            feeds={x:batch_xs_noisy,y:batch_xs,dropout_keep_prob:0.5}
            sess.run(l1_optm,feed_dict=feeds)
            total_cost+=sess.run(l1_cost,feed_dict=feeds)
        if epoch%display==0:
            print('stage1 loss=',total_cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_cost=0.
    for epoch in range(epochs):
        num_batch=int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            l1_h=sess.run(l1_out,feed_dict={x:batch_xs,y:batch_xs,dropout_keep_prob:1.})
            _,l2_cost=sess.run([optm2,l2_cost],feed_dict={l2x:l1_h,l2y:l1_h})
            total_cost+=l2_cost
    print('stage2 ok')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_cost=0.
    for epoch in range(epochs):
        num_batch=int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            l1_h=sess.run(l1_out,feed_dict={x:batch_xs,y:batch_ys,dropout_keep_prob:1.0})
            l2_h=sess.run(l2_out,feed_dict={l2x:l1_h,l2y:l1_h})
            sess.run(l3_optm,feed_dict={l3x:l2_h,l3y:batch_ys})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        num_batch = int(mnist.train.num_examples / batch_size)
        total_cost=0.
        for i in range(num_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            feeds={x:batch_xs,l3y:batch_ys}
            sess.run(optm3,feed_dict=feeds)

