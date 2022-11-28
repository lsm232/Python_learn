import random
import numpy as np

class expand(object):
    def __init__(self,mean):
        self.mean=mean
    def forward(self,image,boxes,labels):
        if random.randint(2):
            return image,boxes,labels
        ratio=random.uniform(1,4)
        h,w,c=image.shape
        expand_img=np.zeros(int(h*ratio),int(w*ratio),c)
        expand_img[:,:,:]=self.mean
        left=random.uniform(0,ratio*w-w)
        top=random.uniform(0,ratio*h-h)
        expand_img[int(top):int(top+h),int(left):int(left+w),:]=image
        image=expand_img

        boxes=boxes.copy()
        boxes[:,:2]+=[int(left),int(top)]
        boxes[:,2:]+=[int(left),int(top)]
        return image,boxes,labels



