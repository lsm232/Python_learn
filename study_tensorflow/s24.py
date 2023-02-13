import numpy as np
import tensorflow as tf
from study_tensorflow.cifar10 import *
import pylab

def load_data(train):
    if train:
        data_dir=r'J:\play\cifar-10-batches-py\data_batch_1'
    else:
        data_dir=r'J:\play\cifar-10-batches-py\test_batch'

    bytestream=open(data_dir,'rb')
    buf=bytestream.read(10000*(1+32*32*3))
    bytestream.close()

    data=np.frombuffer(buf,dtype=np.uint8)
    data=data.reshape(10000,1+32*32*3)
    labels_images=np.hsplit(data,[1])
    labels=labels_images[0].reshape(10000)
    images=labels_images[1].reshape(10000,32,32,3)

    images=images.reshape(10000,3,32,32)
    images=images.transpose(0,2,3,1)

    return images,labels

def weight_v(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def weight_b(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,24,24,3])
y=tf.placeholder(tf.float32,[None,10])
W_conv1=weight_v([5,5,3,64])
b_conv1=weight_b([64])
#x_image=tf.reshape(x,[-1,24,24,3])
h_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
W_conv2=weight_v([5,5,64,64])
b_conv2=weight_b([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
W_conv3=weight_v([5,5,64,10])
b_conv3=weight_b([10])
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
nt_hpool3=avg_pool_6x6(h_conv3)
nt_hpool3_flat=tf.reshape(nt_hpool3,[-1,10])
y_conv=tf.nn.softmax(nt_hpool3_flat)

cross_entropy=-tf.reduce_sum(y*tf.log(y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_pred,'float'))


imgs,labels=load_data(True)




