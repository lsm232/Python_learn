import tensorflow as tf

x=tf.random_normal([4,4],dtype=tf.float32)
b=tf.random_crop(x,[2,2])
sess=tf.Session()
a1=sess.run(x)
a2=sess.run(b)
sess.close()
c=1