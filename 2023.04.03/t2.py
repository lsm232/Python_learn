import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab



def read_bin(order):
    train_path='J:\play\cifar-10-batches-py'+'/'+'data_batch_'+str(order)+'.bin'
    test_path='J:\play\cifar-10-batches-py'+'/'+'test_batch.bin'

    bytestream_train=open(train_path,'rb')
    buf_train=bytestream_train.read(5000*(1+32*32*3))
    bytestream_train.close()

    train_data=np.frombuffer(buf_train,dtype=np.uint8)
    train_data=train_data.reshape(5000,1+32*32*3)
    labels_imgs_train=np.hsplit(train_data,[1])
    train_labels=labels_imgs_train[0].reshape(5000)
    train_imgs=labels_imgs_train[1].reshape(5000,32,32,3)

    bytestream_test = open(test_path, 'rb')
    buf_test = bytestream_test.read(5000 * (1 + 32 * 32 * 3))
    bytestream_test.close()

    test_data = np.frombuffer(buf_test, dtype=np.uint8)
    test_data = test_data.reshape(5000, 1 + 32 * 32 * 3)
    labels_imgs_test = np.hsplit(test_data, [1])
    test_labels = labels_imgs_test[0].reshape(5000)
    test_imgs = labels_imgs_test[1].reshape(5000, 3,32,32)

    return train_labels,train_imgs,test_labels,test_imgs

_,_,_,imgs=read_bin(1)
pylab.imshow(imgs[51].transpose(1,2,0))
pylab.show()
c=1

