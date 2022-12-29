import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tx=np.linspace(0,10,100)
ty=2*tx+np.random.rand(*tx.shape)

X=tf.placeholder('float')
Y=tf.placeholder('float')
w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros(1),name='bias')
z=tf.multiply(w,X)+b

loss=tf.reduce_mean(tf.square(z-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init=tf.global_variables_initializer()
epochs=20
display_freq=2

with tf.Session() as sess:
    sess.run(init)
    plotdata={'batchsize':[],'loss':[]}
    for epoch in range(epochs):
        for (x,y) in zip(tx,ty):
            sess.run(optimizer,feed_dict={X:x,Y:y})

            if epoch%display_freq==0:
                los=sess.run(loss,feed_dict={X:tx,Y:ty})
                print('epoch:',epoch+1,'cost:',los,'w:',sess.run(w),'b=',sess.run(b))
                if not (los=='NA'):
                    plotdata['batchsize'].append(epoch)
                    plotdata['loss'].append(los)

    print('ok')
