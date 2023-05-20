import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist')

tf.reset_default_graph()

#超参
batch_size=128
num_class=10
num_con=2
rand_dim=8
n_input=784
epochs=10
display=1


def generator(x):
    """
    x:(b,class_dim+rand_dim+con_dim)
    output:(b,28,28,1)
    """
    reuse=len([t for t in tf.global_variables() if t.name.startswith('generator')])>0
    with tf.variable_scope('generator',reuse=reuse):
        x=slim.fully_connected(x,num_outputs=256)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        x=slim.fully_connected(x,num_outputs=7*7*32)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        x=tf.reshape(x,(-1,7,7,32))
        x=slim.conv2d_transpose(x,num_outputs=16,kernel_size=(3,3),stride=2)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        x=slim.conv2d_transpose(x,num_outputs=1,kernel_size=(3,3),stride=2)
    return x

def leaky_relu(x):
    return tf.where(tf.greater(x,0),x,0.01*x)

def discriminator(x,num_class,num_cont):
    """
    x:(b,28,28,1)
    output: disc(b,1),recog_cat(b,num_class),recog_con(b,num_cont)
    """
    reuse=len([t for t in tf.global_variables() if t.name.startswith('discriminator')])>0
    with tf.variable_scope('discriminator',reuse=reuse):
        x = slim.conv2d(x, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME', activation_fn=leaky_relu)
        x = slim.conv2d(x, num_outputs=128, kernel_size=(4, 4), stride=2, padding='SAME', activation_fn=leaky_relu)
        flatten_x=slim.flatten(x)
        #real or fake
        disc=slim.fully_connected(flatten_x,num_outputs=1)
        
        #class
        recog_cat=slim.fully_connected(flatten_x,num_outputs=num_class)
        #con
        recog_con=slim.fully_connected(flatten_x,num_outputs=num_cont)
    return disc,recog_cat,recog_con

x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.int32,[None])

z_con=tf.random_normal([batch_size,num_con])
z_rand=tf.random_normal([batch_size,rand_dim])
z=tf.concat([z_con,z_rand,tf.one_hot(y,depth=num_class)],axis=1)
gen=generator(z)

disc_real,class_real,_=discriminator(tf.reshape(x,(-1,28,28,1)),num_class,num_con)
disc_fake,class_fake,con_fake=discriminator(gen,num_class,num_con)

#discirminator loss
real_labels=tf.ones((batch_size))
fake_labels=tf.zeros((batch_size))
loss_d_r=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(disc_real,-1),labels=real_labels))
loss_d_f=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(disc_fake,-1),labels=fake_labels))
loss_d=(loss_d_r+loss_d_f)/2
#g loss
loss_g=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,labels=real_labels))
#class loss
loss_cr=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_real,labels=y))
loss_cf=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_fake,labels=y))
#con loss
loss_con=tf.reduce_mean(tf.square(con_fake-z_con))

t_vars=tf.trainable_variables()
d_vars=[t for t in t_vars if t.name.startswith('discriminator')]
g_vars=[t for t in t_vars if t.name.startswith('generator')]

disc_global_step=tf.Variable(0,trainable=False)
gen_global_step=tf.Variable(0,trainable=False)

train_disc=tf.train.AdamOptimizer(0.0001).minimize(loss_g+loss_cf+loss_con,var_list=g_vars,global_step=gen_global_step)
train_gen=tf.train.AdamOptimizer(0.0001).minimize(loss_d_f+loss_d_r+loss_cf+loss_cr,var_list=d_vars,global_step=disc_global_step)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx,batchy=mnist.train.next_batch(batch_size)
            feeds={x:batchx,y:batchy}
            l_disc,_,l_d_step=sess.run([loss_d,train_disc,disc_global_step],feeds)
            l_gen,_,l_g_step=sess.run([loss_g,train_gen,gen_global_step],feeds)
        if epoch%display==0:
            print('gen loss=',loss_g,'disc loss=',loss_d,'step=',l_g_step)
    print('ok')










