import tensorflow as tf

strps_hosts='localhost:1681'
strworker_hosts='localhost:1681,localhost:1683'


strobj_name='ps'
task_index=0
ps_hosts=strps_hosts.split(',')
worker_hosts=strworker_hosts.split(',')
cluster_spec=tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
server=tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},job_name=strobj_name,task_index=task_index)