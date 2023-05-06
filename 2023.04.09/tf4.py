import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r'G:\leran to play\mnist',one_hot=True)

n_input=28
n_steps=28
n_hidden=128
n_classes=10

x=tf.placeholder(tf.float32,[None,n_steps,n_input])
x1=tf.unstack(x,n_steps,1)
y=tf.placeholder(tf.float32,[None,n_classes])

gru_cell=tf.contrib.rnn.GRUCell(n_hidden)
outputs=tf.contrib.rnn.static_rnn(gru_cell,x1,dtype=tf.float32)
pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn=None)


lr_rate=0.001
iters=10000
batch_size=32
display=10

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=1
    while step*batch_size<iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(-1,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display==0:
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print('loss=',loss)
        step+=1
    print('ok')

    test_len=128
    test_data=mnist.test.images[:test_len].reshape(-1,n_steps,n_input)
    test_label=mnist.test.labels[:test_len]
    print('test acc:{:.4f}'.format(sess.run(acc,feed_dict={x:test_data,y:test_label})))


