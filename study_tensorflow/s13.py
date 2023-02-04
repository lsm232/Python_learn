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
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


epochs=25
batch_size=4
display_freq=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=c/batch_size

            if (i+1)%display_freq==0:
                print("epoch:",'%04d'%(epoch+1),'cost=%0.9f'%avg_cost)
    print('fnished')

    test_total_batch=int(mnist.test.num_examples/batch_size)
    for i in range(test_total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)

        correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax())