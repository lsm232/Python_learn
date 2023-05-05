import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)



n_input=28
n_steps=28
n_hidden=128
n_classes=10
x=tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
x1=tf.unstack(x,28,1)
lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
outputs,states=tf.contrib.rnn.static_rnn(lstm_cell,x1,dtype=tf.float32)
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn=None)

cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
c=1