from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

tf.reset_default_graph()
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))

pred=tf.nn.softmax(tf.matmul(x,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


