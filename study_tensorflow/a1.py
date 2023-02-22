import tensorflow as tf

# hello=tf.constant('hello world')
# sess=tf.Session()
# print(sess.run(hello))
# sess.close()

# with tf.Session() as sess:
#     a=tf.constant(1)
#     b=tf.constant(2)
#     print(sess.run(a+b))

a=tf.placeholder(dtype=tf.int8)
b=tf.placeholder(dtype=tf.int8)
c=a+b
with tf.Session() as sess:
    print(sess.run(c,feed_dict={a:1,b:2}))
