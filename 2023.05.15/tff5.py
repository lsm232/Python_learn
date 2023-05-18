import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

tf.reset_default_graph()

def generator(x):
    reuse=len([t for t in tf.global_variables() if t.name.startwith('generator')])>0
    with tf.variable_scope('generator',reuse=reuse):
        x=slim.fully_connected(x,1024)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        x=slim.fully_connected(x,7*7*128)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        x=tf.reshape(x,[-1,7,7,128])
        x=slim.conv2d_transpose(x,64,kernel_size=(4,4),stride=2)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        z=slim.conv2d_transpose(x,1,kernel_size=(4,4),stride=2,activation_fn=tf.nn.sigmoid)
    return z

def leaky_relu(x):
    return tf.where(tf.greater(x,0),x,x*0.01)

def discriminator(x,num_classes=10,num_cont=2):
    reuse=len([t for t in tf.global_variables() if t.name.startwith('discriminatore')])>0
    with tf.variable_scope('discriminator',reuse):
        x=tf.reshape(x,shape=(-1,28,28,1))
        x=slim.conv2d(x,num_outputs=64,kernel_size=[4,4],stride=2,activation_fn=leaky_relu)
        x=slim.conv2d(x,num_outputs=128,kernel_size=[4,4],stride=2,activation_fn=leaky_relu)
        x=slim.flatten(x)
        shared_tensor=slim.fully_connected(x,num_outputs=1024,activation_fn=leaky_relu)
        recog_shared=slim.fully_connected(shared_tensor,num_outputs=128,activation_fn=leaky_relu)
        disc=slim.fully_connected(shared_tensor,num_outputs=1)
        disc=tf.squeeze(disc,-1)
        recog_cat=slim.fully_connected(recog_shared,num_outputs=num_classes,activation_fn=None)
        recog_cont=slim.fully_connected(recog_shared,num_outputs=num_cont,activation_fn=tf.nn.sigmoid)
    return disc,recog_cat,recog_cont

batch_size=10
classes_dim=10
con_dim=2
rand_dim=38
n_input=784

x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.int32,[None])
z_con=tf.random_normal((batch_size,con_dim))
z_rand=tf.random_normal((batch_size,rand_dim))
z=tf.concat(axis=1,values=[tf.one_hot(y,depth=classes_dim),z_con,z_rand])
gen=generator(z)
genout=tf.squeeze(gen,-1)
y_real=tf.ones(batch_size)
y_fake=tf.zeros(batch_size)
disc_real,class_real,_=discriminator(x)
disc_fake,class_fake,con_fake=discriminator(gen)
pred_class=tf.argmax(class_fake,dimension=1)
