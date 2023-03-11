import tensorflow as tf

trainX=tf.linspace(-1,1,100)
trainY=0.5*trainX+tf.random_normal([1])

strps_hosts='localhost:1681'
strworkers_hosts='localhost:1682,localhost:1683'

ps_hosts=strps_hosts.split(',')
worker_hosts=strworkers_hosts.split(',')

cluster_spec=tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
sever=tf.train.Server(
    {'ps':ps_hosts,'worker':worker_hosts},
    job_name='ps',
    task_index=0
)

print('wait')
sever.join()

with tf.device(
        tf.train.replica_device_setter(worker_device='/job:worker/task:%d'%0,cluster=cluster_spec),
):
    X=tf.placeholder('float')
    Y=tf.placeholder('float')
    W=tf.Variable(tf.random_normal([1]),name='weight')
    b=tf.Variable(tf.zeros([1]),name='bias')
    z=tf.multiply(X,W)+b

    global_step=tf.train.get_or_create_global_step()
    tf.summary.histogram('z',z)

    cost=tf.reduce_mean(tf.square(z-Y))
    optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(cost,global_step=global_step)

    tf.summary.scalar('loss_fn',cost)

    saver=tf.train.Saver(max_to_keep=1)
    merged_summary_op=tf.summary.merge_all()

    init=tf.global_variables_initializer()

    epochs=2200
    display=2

    sv=tf.train.Supervisor(
        is_chief=(True),
        logdir='log/super/',
        init_op=init,
        summary_op=None,
        saver=saver,
        global_step=global_step,
        save_model_secs=10,
    )
    with sv.managed_session(sever.target) as sess:
        print('sess ok')
        print(global_step.eval(session=sess))
        for epoch in range(global_step.eval(session=sess),epochs*len(trainX)):
            for (x,y) in zip(trainX,trainY):
                _,epoch=sess.run([optimizer,global_step],feed_dict={X:x,Y:y});
                summary_str=sess.run(merged_summary_op,feed_dict={X:x,Y:y})
                sv.summary_computed(sess,summary_str,global_step=global_step)
                
