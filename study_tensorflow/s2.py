import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata={'batch_size':[],'loss':[]}
def moving_average(a,w=10):
    if len(a)<10:
        return a
    else:
        return [val if idx<w else sum(a[(idx-w):idx]) for idx,val in enumerate(a)]

trainx=np.linspace(-1,1,100)
trainy=trainx*2+np.random.rand(*trainx.shape)

plt.plot(trainx,trainy,'ro',label='train data')
plt.legend()
plt.show()


#https://blog.csdn.net/Fwuyi/article/details/125637504
tf.reset_default_graph()

X=tf.placeholder('float')
Y=tf.placeholder('float')
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros(1),name='bias')
z=tf.reduce_mean(tf.multiply(X,W)+b)
tf.summary.scalar('z',z)

loss=tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_fn',loss)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init=tf.global_variables_initializer()
epochs=20
display_freq=2
saver=tf.train.Saver(max_to_keep=1)
saverdir='./log'

with tf.Session() as sess:
    sess.run(init)
    merged_summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter('log/mnist_',sess.graph)
    for epoch in range(epochs):
        for (x,y) in zip(trainx,trainy):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if epoch%display_freq==0:
            los=sess.run(loss,feed_dict={X:trainx,Y:trainy})
            print('epoch:',epoch+1,'loss:',los,'W:',sess.run(W),'b:',sess.run(b))
            if los!='NA':
                plotdata['batch_size'].append(epoch)
                plotdata['loss'].append(los)
            saver.save(sess,saverdir+'/'+"model_.ckpt",global_step=epoch)
        summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
        summary_writer.add_summary(summary_str, epoch)
    print('finished')
    print('cost:',sess.run(loss,feed_dict={X:trainx,Y:trainy}),'W:',sess.run(W),'b:',sess.run(b))


    plt.plot(trainx,trainy,'ro',label='data')
    plt.plot(trainx,sess.run(W)*trainx+sess.run(b),label='fit')
    plt.legend()
    plt.show()

    plotdata['avgloss']=moving_average(plotdata['loss'])
    plt.figure(1)
    plt.plot(plotdata['batch_size'],plotdata['avgloss'],'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.show()

load_epoch=18
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,saverdir+'/'+'model_.ckpt-'+str(load_epoch))
    print('x=0.2,z=',sess2.run(z,feed_dict={X:0.2}))

