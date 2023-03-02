import tensorflow as tf
import numpy as np

x=tf.constant(2)
y=tf.constant(5)
def f1():
    return tf.multiply(x,17)
def f2():
    return tf.add(y,23)
r=tf.cond(tf.less(x,y),f1,f2)
sess=tf.Session()
print(sess.run(r))
sess.close()