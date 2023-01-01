import tensorflow as tf

strps_hosts='localhost:1681'
strworker_hosts='localhost:1682,localhost:1683'

strjob_name='ps'
task_index=0
ps_hosts=strps_hosts.split(',')
worker_hosts=strworker_hosts.split(',')

cluster_spec=tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
server=tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},job_name=strjob_name,task_index=task_index)

if strjob_name=='ps':
    print('wait')
    server.join()

with tf.device(tf.train.replica_device_setter(worker_device='./job:worker/task:%d'%task_index,cluster=cluster_spec)):
    X=tf.placeholder('float')
    Y=tf.placeholder('float')
    W=tf.Variable(tf.random_normal([1]),name='weight')
    b=tf.Variable(tf.zeros([1]),name='bias')

    global_step=tf.train.get_or_create_global_step()
    z=tf.multiply(X,W)+b
    tf.summary.histogram('z',z)
    cost=tf.reduce_mean(tf.square(Y-z))
    tf.summary.scalar('loss_function',cost)
    learing_rate=0.01
    optimizer=tf.train.GradientDescentOptimizer(learing_rate).minimize(cost,global_step=global_step)
    saver=tf.train.Saver(max_to_keep=1)
    merged_summary_op=tf.summary.merge_all()
    init=tf.global_variables_initializer()