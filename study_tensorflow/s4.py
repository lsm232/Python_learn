import tensorflow as tf

c=tf.get_default_graph()
print(c)
with tf.Graph().as_default():
    print(tf.get_default_graph())