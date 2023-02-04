import tensorflow as tf

global_step=tf.Variable(5,trainable=False)
init_lr_rate=0.1
learing_rate=tf.train.exponential_decay(init_lr_rate,global_step=global_step,decay_steps=10,decay_rate=0.9)
opt=tf.train.GradientDescentOptimizer(learing_rate)
add_global=global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learing_rate))
    for i in range(20):
        g,rate=sess.run([add_global,learing_rate])
        print(g,rate)
