import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)
train_imgs=data.train.images
train_labels=data.train.labels
test_imgs=data.test.images
test_labels=data.test.labels
print(train_imgs.shape)

#train opts
epochs=25
batch_size=8
display_freq=2
lr=0.001
save_path=r'log/t1model.ckpt'

X=tf.placeholder(dtype=tf.float32,shape=[None,784])
Y=tf.placeholder(dtype=tf.float32,shape=[None,10])
w1=tf.Variable(tf.random_normal([784,256]))
w2=tf.Variable(tf.random_normal([256,256]))
w3=tf.Variable(tf.random_normal([256,10]))
b1=tf.Variable(tf.zeros([256]))
b2=tf.Variable(tf.zeros([256]))
b3=tf.Variable(tf.zeros([10]))
out1=tf.nn.relu(tf.matmul(X,w1)+b1)
out2=tf.nn.relu(tf.matmul(out1,w2)+b2)
pred=tf.matmul(out2,w3)+b3

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(Y,1)),tf.float32))
optimizer=tf.train.AdamOptimizer(lr).minimize(cost)
saver=tf.train.Saver()

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iters=int(train_labels.shape[0]/batch_size)
    for epoch in range(epochs):
        avg_loss=0.
        for iter in range(iters):
            x,y=data.train.next_batch(batch_size)
            _,loss=sess.run([optimizer,cost],feed_dict={X:x,Y:y})
            avg_loss=avg_loss+loss/iters
        if (epoch+1)%display_freq==0:
            print('epoch:',epoch+1,"loss:",avg_loss)

    #save model
    saver.save(sess,save_path)

    #test
    print('train finished!')
    print('acc:',acc.eval({X:test_imgs,Y:test_labels}))


