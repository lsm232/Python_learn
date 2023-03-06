import tensorflow as tf

# with tf.variable_scope('test1'):
#     var1=tf.get_variable(name='var21',shape=[1],dtype=tf.float32)
# with tf.variable_scope('test2'):
#     var1=tf.get_variable(name='var21',shape=[1,2],dtype=tf.float32)
# print(var1)
# with tf.variable_scope('test2',reuse=True):
#     var2=tf.get_variable('var21')
# print(var2)

with tf.variable_scope('t1',initializer=tf.constant_initializer(0.4)):
    var1=tf.get_variable('var1',shape=[3])
    var2=tf.get_variable('var2',shape=[3],dtype=tf.float32,initializer=tf.constant_initializer(0.2))
    with tf.variable_scope('t2',initializer=tf.constant_initializer(0.1)):
        var3=tf.get_variable('var3',shape=[3])
        var4=tf.get_variable('var4',shape=[3],initializer=tf.constant_initializer(0.))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.eval())
    print(var2.eval())
    print(var3.eval())
    print(var4.eval())
