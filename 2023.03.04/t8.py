import tensorflow as tf

strps_hosts='localhost:1681'
strworkers_hosts='localhost:1682,localhost:1683'

ps_hosts=strps_hosts.split(',')
worker_hosts=strworkers_hosts.split(',')
c=1

task_id=0
task_name='ps'