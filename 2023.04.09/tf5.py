import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

#超参
batch_size=32
iters=10000
lr_rate=0.001
display=10

#网络相关的参数
n_steps=28
n_input=28
n_hidden=128
n_classes=10

#构建网络
x=tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
gru_cell=tf.contrib.rnn.GRUCell(n_hidden)
outputs,_=tf.nn.dynamic_rnn(gru_cell,x,dtype=tf.float32)
outputs=tf.transpose(outputs,[1,0,2])
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(lr_rate).minimize(cost)


#评估网络
correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct,tf.float32))


#开始训练
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size<iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(-1,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if  step%display==0:
            print('loss=',sess.run(cost,feed_dict={x:batch_x,y:batch_y}))
        step+=1
    print('ok')

    test_len=128
    test_data=mnist.test.images[:test_len].reshape(test_len,n_steps,n_input)
    test_label=mnist.test.labels[:test_len]
    print('acc={:.4f}'.format(sess.run(acc,feed_dict={x:test_data,y:test_label})))
