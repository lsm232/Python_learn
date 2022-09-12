import random
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms
    def __call__(self,image,target):
        for transform in self.transforms:
            image,target=transform(image,target)
        return image,target

class Totensor(object):
    def __call__(self, img,target):
        return F.to_tensor(img),target

class RandomHorizontalFlip(object):
    def __init__(self,prob=0.5):
        self.prob=prob
    def __call__(self, image,target):
        if self.prob<random.random():
            image=image.flip(-1)
            height,width=image.shape[-2:]
            bbox=target['boxes']
            bbox[:,[0,2]]=width-bbox[:,[2,0]]
            target['boxes']=bbox
        return image,target

data_transforms={'train':Compose([Totensor(),RandomHorizontalFlip(0.9)]),'test':Compose([Totensor()])}