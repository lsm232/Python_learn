from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf

mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)
# print('输入数据的shape:',mnist.train.images.shape)
# print('输入数据的label:',mnist.train.labels.shape)
# im=mnist.train.images[1].reshape(-1,28)
# pylab.imshow(im)
# pylab.show()
tf.reset_default_graph()
x=tf.placeholder(tf.float32,shape=(None,784))
y=tf.placeholder(tf.float32,shape=(None,10))
W=tf.Variable(tf.random_normal(shape=[784,10]))
b=tf.Variable(tf.zeros(shape=[10]))

pred=tf.nn.softmax(tf.matmul(x,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs=25
batch_size=100
display_step=1
saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=c/total_batch
        if (epoch+1)%display_step==0:
            print('epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost))
    print('finished!')

    save_path=r'./checkpoints/0101.ckpt'
    save_path=saver.save(sess,save_path)
