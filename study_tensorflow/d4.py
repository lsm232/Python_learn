import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata={'batchsize':[],'loss':[]}

def moving_average(a,w=10):
    if len(a)<w:
        return a
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)
# plt.plot(train_X,train_Y,'ro',label='origin data')
# plt.legend()
# plt.show()
X=tf.placeholder(dtype=tf.float32)
Y=tf.placeholder(dtype=tf.float32)
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
z=tf.multiply(W,X)+b
cost=tf.reduce_mean(tf.square(z-Y))
lr_rate=0.001
optimizer=tf.train.GradientDescentOptimizer(lr_rate).minimize(cost)

init=tf.global_variables_initializer()
epochs=20
display_step=2
saver=tf.train.Saver(max_to_keep=1)
save_path='log/'

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for x,y in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if (epoch+1)%display_step==0:
            loss=sess.run(cost,feed_dict={X:x,Y:y})
            print('epoch=',epoch,'loss=',loss,'w=',sess.run(W),'b=',sess.run(b))
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
            saver.save(sess,save_path=save_path+'model1.cpkt',global_step=epoch)
    print('train finished!')
    print('epoch=', epoch, 'loss=', loss, 'w=', sess.run(W), 'b=', sess.run(b))

    plt.plot(train_X,train_Y,'ro',label='data')
    plt.plot(train_X,train_X*sess.run(W)+sess.run(b),label='fit line')
    plt.legend()
    plt.show()

    plotdata['avgloss']=moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'],plotdata['avgloss'],'b--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('epoch/loss')
    plt.show()

load_epoch=19
with tf.Session() as sess2:
    sess2.run(init)
    saver.restore(sess2,save_path+'model1.cpkt-'+str(load_epoch))
    print('x=0.2,z=',sess2.run(z,feed_dict={X:0.2}))

c=1