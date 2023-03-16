import random
import numpy as np
import torch

class Expand(object):
    def __init__(self,mean):
        self.mean=mean
    def __call__(self,img,boxes):
        h,w,c=img.shape
        if random.randint(0,1):
            return img
        ratio=random.uniform(1,4)
        expand_img=np.zeros([ratio*h,ratio*w,c],dtype=img.dtype)
        expand_img[:,:,:]=self.mean
        left=random.uniform(0,ratio*w-w)
        top=random.uniform(0,ratio*h-h)
        expand_img[top:top+h,left:left+w]=img

        boxes=boxes.copy()
        boxes[:,:2]+=[top,left]
        boxes[:,2:]+=[top,left]
        return img,boxes
