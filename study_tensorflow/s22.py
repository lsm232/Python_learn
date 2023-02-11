import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from .cifar10 import *
import pylab

batch_size=128
data_dir=r'J:\play\cifar-10-python'
images_test,labels_test=inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)


