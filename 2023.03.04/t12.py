import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

epochs=20
lr_rate=0.002
display_freq=2
batch_size=8


X=tf.placeholder(dtype=tf.float32,shape=[None,784])
Y=tf.placeholder(dtype=tf.float32,shape=[None,10])
W1=tf.Variable(tf.truncated_normal([784,10],stddev=0.01))
b1=tf.Variable(tf.zeros([10]))
z1=tf.matmul(X,W1)+b1
maxout=tf.reduce_max(z1,axis=1,keep_dims=True)
W2=tf.Variable(tf.truncated_normal([1,10],stddev=0.01))
b2=tf.Variable(tf.zeros([10]))
pred=tf.nn.softmax(tf.matmul(maxout,W2)+b2)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=-1))
optimizer=tf.train.GradientDescentOptimizer(lr_rate).minimize(cost)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_loss=0.
        total_batch=int(len(mnist.train.labels)/batch_size)
        for i in range(total_batch):
            x,y=mnist.train.next_batch(batch_size)
            _,loss=sess.run([optimizer,cost],feed_dict={X:x,Y:y})
            avg_loss+=loss/total_batch

        if (epoch+1)%display_freq==0:
            print('epoch:',epoch+1,'loss={:.4f}'.format(avg_loss))

