import numpy as np
import pylab


filename=r'J:\play\cifar-10-batches-py\data_batch_5'
bytestream=open(filename,'rb')
buf=bytestream.read(10000*(1+32*32*3))
bytestream.close()

data=np.frombuffer(buf,dtype=np.uint8)
data=data.reshape(10000,1+32*32*3)
labels_images=np.hsplit(data,[1])
labels_=labels_images[0].reshape(10000)
images_=labels_images[1].reshape(10000,32,32,3)

# img=np.reshape(images_[3200],(3,32,32))
# img=img.transpose(1,2,0)
pylab.imshow(images_[5200])
pylab.show()
