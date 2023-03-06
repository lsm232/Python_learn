
import tensorflow as tf

get_var1=tf.get_variable('firstvar',[1],initializer=tf.constant_initializer(0.3))
sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(get_var1.eval())
sess.close()