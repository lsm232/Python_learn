import tensorflow as tf

# a=tf.constant(1)
# g=tf.Graph()
# with g.as_default():
#     b=tf.constant(2)
#     print(a.graph)
#     print(b.graph)
# g2=tf.get_default_graph()
# print(g2)
# tf.reset_default_graph()
# g3=tf.get_default_graph()
# print(g3)
# print(b.name)
# t=g.get_tensor_by_name('Const:0')
# print(t)

# a=tf.constant([[1.,2.]])
# # b=tf.constant([[1.],[2.]])
# # c=tf.multiply(a,b,name='op1')
# # print(c)
# # print(c.name)
# # d=tf.get_default_graph().get_tensor_by_name('op1:0')
# # print(d)
# # print(c.op.name)
# # print(tf.get_default_graph().get_operation_by_name('op1'))
# #
# # with tf.Session() as sess:
# #     print(sess.run(c))
# #
# # print(tf.get_default_graph().get_operations())
# # print(tf.get_default_graph().as_graph_element(c))

print(tf.get_default_graph())
with tf.Graph().as_default():
    print(tf.get_default_graph())