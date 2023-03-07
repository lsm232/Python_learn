import tensorflow as tf

with tf.variable_scope('test1'):
    with tf.name_scope('bar1'):
        v=tf.get_variable('v',shape=[1])
        x=v+0.1
print(v.name)
print(x.op.name)
print(x.name)