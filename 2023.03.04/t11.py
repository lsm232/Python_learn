import tensorflow as tf

global_step=tf.Variable(0,False)
add_step=global_step.assign_add(1)
lr_rate=1.
lr_rate_decay=tf.train.exponential_decay(lr_rate,global_step,decay_steps=10,decay_rate=0.5)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1,12):
        lr,step=sess.run([lr_rate_decay,add_step])
        print(lr,step)
