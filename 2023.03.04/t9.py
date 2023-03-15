from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf

mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

# imgs=mnist.train.images
# img=imgs[1].reshape(-1,28)
# pylab.imshow(img)
# pylab.show()
# c=1

epochs=25
batch_size=8
display_freq=2

X=tf.placeholder(dtype=tf.float32,shape=[None,28*28])
Y=tf.placeholder(dtype=tf.float32,shape=[None,10])

W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))
pred=tf.nn.softmax(tf.matmul(X,W)+b)
#pred=tf.matmul(X,W)+b

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
correct_prediction=tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(cost)

init=tf.global_variables_initializer()
saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     total_batch=int(mnist.train.images.shape[0]/batch_size)
#     for epoch in range(epochs):
#         avg_cost=0.
#         for i in range(total_batch):
#             batchx,batchy=mnist.train.next_batch(batch_size)
#             _,c=sess.run([optimizer,cost],feed_dict={X:batchx,Y:batchy})
#             avg_cost+=c/total_batch
#         if (epoch+1)%display_freq==0:
#             print("epoch=%d"%(epoch+1),'cost={:.4f}'.format(avg_cost))
#     print('ok')
#     print('acc=',acc.eval({X:mnist.validation.images,Y:mnist.validation.labels}))
#
#     save_path=saver.save(sess,'log/model1.ckpt')

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'./log/model1.ckpt')
    print('acc=',sess.run([acc],feed_dict={X:mnist.validation.images,Y:mnist.validation.labels}))

    output=tf.argmax(pred,1)
    batch_xs,batch_ys=mnist.train.next_batch(2)
    outputval=sess.run([output],feed_dict={X:batch_xs})
    print('pred:',outputval,'label:',batch_ys)

    pylab.imshow(batch_xs[1].reshape(-1,28))
    pylab.show()