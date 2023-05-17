import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)
from scipy.stats import norm

n_input=784
n_hidden_1=256
n_hidden_2=2
x=tf.placeholder(tf.float32,[None,n_input])
zinput=tf.placeholder(tf.float32,[None,n_hidden_2])

weights={
    'w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1],stddev=0.001)),
    'b1':tf.Variable(tf.zeros([n_hidden_1])),
    'mean_w1':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],stddev=0.001)),
    'mean_b1':tf.Variable(tf.zeros([n_hidden_2])),
    'log_sigma_w1':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],stddev=0.001)),
    'log_sigma_b1':tf.Variable(tf.zeros([n_hidden_2])),
    'w2':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_1],stddev=0.001)),
    'b2':tf.Variable(tf.zeros([n_hidden_1])),
    'w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.001)),
    'b3': tf.Variable(tf.zeros([n_input])),
}

h1=tf.nn.relu(tf.add(tf.matmul(x,weights['w1']),weights['b1']))
z_mean=tf.add(tf.matmul(h1,weights['mean_w1']),weights['mean_b1'])
z_log_sigma_sq=tf.add(tf.matmul(h1,weights['log_sigma_w1']),weights['log_sigma_b1'])

eps=tf.random_normal(tf.stack([tf.shape(h1)[0],n_hidden_2]),0,1,dtype=tf.float32)
z=tf.add(z_mean,tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)),eps))
h2=tf.add(tf.matmul(z,weights['w2']),weights['b2'])
reconstruction=tf.matmul(h2,weights['w3'])+weights['b3']
h2out=tf.nn.relu( tf.matmul(zinput, weights['w2'])+ weights['b2'])
reconstructionout = tf.matmul(h2out, weights['w3'])+ weights['b3']
# reconstr_loss=0.5*tf.reduce_mean(tf.pow(reconstruction-x,2.0))
# latent_loss=-0.5*tf.reduce_mean(1+z_log_sigma_sq-tf.square(z_mean)-tf.exp(z_log_sigma_sq))
# cost=reconstr_loss+latent_loss
reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, x), 2.0))
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                   - tf.square(z_mean)
                                   - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

training_epochs=5
batch_size=128
display=1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            _,c,d1,d2=sess.run([optimizer,cost,reconstr_loss,latent_loss],feed_dict={x:batch_x})
        if epoch%display==0:
            print('epoch:',c)
    print('ok')

    show_num=10
    pred=sess.run(reconstruction,feed_dict={x:mnist.test.images[:show_num]})
    f,a=plt.subplots(2,10,figsize=(10,2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(pred[i],(28,28)))
    plt.show()

    n=15
    size=28
    figure=np.zeros((size*n,size*n))
    grid_x,grid_y=norm.ppf(np.linspace(0.05,0.95,n)),norm.ppf(np.linspace(0.05,0.95,n))
    for i,y1 in enumerate(grid_x):
        for j,x1 in enumerate(grid_y):
            z_sample=np.array([[x1,y1]])
            x_decoded=sess.run(reconstructionout,feed_dict={zinput:z_sample})
            digit=x_decoded[0].reshape(28,28)
            figure[i*size:(i+1)*size,j*size:(j+1)*size]=digit
    plt.figure(figsize=(10,10))
    plt.imshow(figure)
    plt.show()


