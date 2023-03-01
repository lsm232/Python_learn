import tensorflow as tf
import numpy as np

trainx=np.linspace(-1,1,100)
trainy=2*trainx+np.random.randn(*trainx.shape)*0.3

x=tf.placeholder(dtype=tf.float32)
y=tf.placeholder(dtype=tf.float32)
w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.zeros([1]))
z=tf.multiply(x,w)+b
tf.summary.histogram('z',z)

cost=tf.reduce_mean(tf.square(y-z))
tf.summary.scalar('loss_fn',cost)
optimizer=tf.train.GradientDescentOptimizer(0.02).minimize(cost)

epochs=20
display_step=2
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged_summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter('log/',sess.graph)
    for epoch in range(epochs):
        for a,m in zip(trainx,trainy):
            sess.run(optimizer,feed_dict={x:a,y:m})
            loss=sess.run(cost,feed_dict={x:a,y:m})
        if (epoch+1)%2==0:
            print('w=',sess.run(w),'b=',sess.run(b),'cost=',loss)
        summary_str=sess.run(merged_summary_op,feed_dict={x:a,y:m})
        summary_writer.add_summary(summary_str,epoch)
    print('train finished')
