import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)



n_input=28
n_steps=28
n_hidden=128
n_classes=10
x=tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
x1=tf.unstack(x,28,1)
lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
outputs,states=tf.contrib.rnn.static_rnn(lstm_cell,x1,dtype=tf.float32)
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn=None)

lr_rate=0.001
training_iters=100000
batch_size=32
display_step=10

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=1
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display_step==0:
            accs=sess.run(acc,feed_dict={x:batch_x,y:batch_y})
            print('acc={:.5f}'.format(accs))
        step+=1
    print('ok')
    test_len=128
    test_data=mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label=mnist.test.labels[:test_len]
    print('acc',sess.run(acc,feed_dict={x:test_data,y:test_label}))