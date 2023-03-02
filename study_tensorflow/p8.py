import tensorflow as tf

x=tf.reshape(tf.linspace(0.,9.,9),(3,3))
b=tf.random_crop(x,[2,2])
sess=tf.Session()
a1=sess.run(x)
a2=sess.run(b)
sess.close()
c=1