import tensorflow as tf

save_dir='/log'
saver=tf.train.Saver()
kpt=tf.train.latest_checkpoint(save_dir)
if kpt!=None:
    saver.restore(sess,kpt)