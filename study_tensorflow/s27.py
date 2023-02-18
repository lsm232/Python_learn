import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

lr_rate=tf.train.exponential_decay(0.04,0,1000,0.9)