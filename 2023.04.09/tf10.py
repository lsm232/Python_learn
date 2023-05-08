import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

#超参
batch_size=32
iters=10000
display=10
lr_rate=0.002

#构建网络
n_input=28
n_steps=28
n_hidden=128
n_classes=10
x=tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
x1=tf.unstack(x,n_steps,1)
fw_rnn=[]
bw_rnn=[]
for i in range(2):
    fw_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0))
    bw_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0))
outputs,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_rnn,bw_rnn,x1,dtype=tf.float32)
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(lr_rate).minimize(cost)
acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),dtype=tf.float32))

#开始训练
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size<iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display==0:
            print('loss=',sess.run(cost,feed_dict={x:batch_x,y:batch_y}))
        step+=1
    print('ok')

    #开始评估
    test_len=200
    test_x=mnist.test.images[:test_len].reshape(test_len,n_steps,n_input)
    test_y=mnist.test.images[:test_len]
    print('acc={:.4f}'.format(sess.run(acc,feed_dict={x:batch_x,y:batch_y})))
