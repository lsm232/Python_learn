import tensorflow as tf
import numpy as np

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
    'h3':tf.Variable(tf.random_normal(n_hidden_2,n_classes))
}
bias={
    'b1':tf.Variable(tf.zeros([n_hidden_1])),
    'b2':tf.Variable(tf.zeros([n_hidden_2])),
    'b3':tf.Variable(tf.zeros([n_classes]))
}


def mlp(x,weights,bias):
    layer_1=tf.nn.relu(tf.add(tf.multiply(x,weights['h1']),bias['b1']))
    layer_2=tf.nn.relu(tf.add(tf.multiply(layer_1,weights['h2']),bias['b2']))
    layer_3=tf.matmul(layer_2,weights['h3'])+bias['b3']
    return layer_3

pred=mlp(x,weights,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(lr_rate).minimize(cost)
