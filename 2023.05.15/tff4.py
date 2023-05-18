import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

#超参
n_labels=10
n_input=784
n_hidden_1=256
n_hidden_2=2
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_labels])
zinput=tf.placeholder(tf.float32,[None,n_hidden_2])
weights={
    'w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1],stddev=0.001)),
    'w2':tf.Variable(tf.truncated_normal([n_labels,n_hidden_1],stddev=0.001)),
    'w3_mean':tf.Variable(tf.truncated_normal([n_hidden_1*2,n_hidden_2],stddev=0.001)),
    'w3_log_sigma':tf.Variable(tf.truncated_normal([n_hidden_1*2,n_hidden_2],stddev=0.001)),
    'w4':tf.Variable(tf.truncated_normal([n_hidden_2+n_labels,n_hidden_1],stddev=0.001)),
    'w5':tf.Variable(tf.truncated_normal([n_hidden_1,n_input],stddev=0.001)),
}
l1_img=tf.nn.relu(tf.matmul(x,weights['w1']))
l1_label=tf.nn.relu(tf.matmul(y,weights['w2']))
l1_img_label=tf.concat([l1_img,l1_label],1)
mean=tf.nn.relu(tf.matmul(l1_img_label,weights['w3_mean']))
log_sigma=tf.nn.relu(tf.matmul(l1_img_label,weights['w3_log_sigma']))
eps=tf.random_normal(tf.stack([tf.shape(l1_img)[0],n_hidden_2]),0,1,dtype=tf.float32)
z=tf.add(tf.sqrt(tf.exp(log_sigma))*eps,mean)
z_label=tf.concat([z,y],1)
out1=tf.nn.relu(tf.matmul(z_label,weights['w4']))
out2=tf.matmul(out1,weights['w5'])
zinputall=tf.concat([zinput,y],1)
output=tf.matmul(tf.nn.relu(tf.matmul(zinputall,weights['w4'])),weights['w5'])

#cost
reconstr_loss=tf.reduce_sum(tf.pow(out2-x,2),1)
latent_loss=-1*tf.reduce_sum(1+log_sigma-tf.square(mean)-tf.exp(log_sigma),1)
cost=tf.reduce_mean(reconstr_loss+latent_loss)
optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)

#
epochs=10
batch_size=128
display=2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_loss=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batchx,batchy=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batchx,y:batchy})
        if epoch%display==0:
            print('loss=',c)
    print('finish')

    test_num=20
    z_sample=np.random.randn(10,2)
    pred=sess.run(output,feed_dict={zinput:z_sample,y:mnist.test.labels[10:test_num]})
    f,a=plt.subplots(2,10,figsize=(10,2))
    for i in range(10):
        a[0][i].imshow(np.reshape(mnist.test.images[10+i],(28,28)))
        a[1][i].imshow(np.reshape(pred[i],(28,28)))
    plt.show()


