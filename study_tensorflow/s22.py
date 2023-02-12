import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from study_tensorflow.cifar10 import *
import pylab

batch_size=128
data_dir=r'J:\play\cifar-10-python'
images_test,labels_test=inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
images_batch,labels_batch=sess.run([images_test,labels_test])
print(images_batch.shape)
print(labels_batch.shape)
c=images_batch[0].max()-images_batch[0].min()
plt.imshow((images_batch[0]-images_batch[0].min())/c)
plt.show()

