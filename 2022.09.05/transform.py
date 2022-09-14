import torch
import torch.nn as nn
from typing import Tuple,List

class normalize(object):
    def __init__(self,mean,std):
        self.mean=mean  #tensor [3]
        self.std=std    #tensor [3]
    def __call__(self, image):
        mean=self.mean[:,None,None]
        std=self.std[:,None,None]
        image=(image-mean)/std
        return image

class resize_image(object):
    def __init__(self,min_size,max_size):  #resize img to min_size-max_size
        self.min_size=min_size
        self.max_size=max_size
    def __call__(self, image):
        h,w=image.shape[-2:]
        min_=min(h,w)
        max_=max(h,w)
        ratio=float(self.min_size[0])/min_

        if ratio*max_>self.max_size:
            ratio=self.max_size/max_
        image=nn.functional.interpolate(image,scale_factor=ratio)
        return image

class resize_boxes(object):
    def __init__(self,origin_size,new_size):
        self.origin_size=origin_size
        self.new_size=new_size
    def __call__(self, target):
        ratios=[torch.tensor(s,dtype=torch.float32)/torch.tensor(s_o,dtype=torch.float32) for s,s_o in zip(self.new_size,self.origin_size)]
        ratio_height,ratio_width=ratios
        boxes=target['boxes']
        xmin,ymin,xmax,ymax=boxes.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        target['boxes']=torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return target

class find_max(object):
    def __call__(self, image_sizes):
        max=image_sizes[0]
        for img_size in image_sizes[1:]:
            for i,item in enumerate(img_size):
                max[i]=max(max[i],item)
        return max

class to_batch_images(object):
    def __init__(self,batch_shape):
        self.batch_shape=batch_shape
    def __call__(self,images):
        batch_images = images[0].new_full(self.batch_shape, fill_value=0)
        for img, batch_img in zip(images, batch_images):
            batch_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        return batch_images

class Imagelist(object):
    def __init__(self,batch_imgs,resized_sizes,origin_size):
        self.batch_imgs=batch_imgs
        self.resized_sizes=resized_sizes
        self.origin_size=origin_size

class postprocess(object):
    def __call__(self, result,image_shapes,origin_shapes):
        if self.training:
            return result
        for i,(pred,image_shape,origin_shape) in enumerate(zip(result,image_shapes,origin_shapes)):
            boxes=pred['boxes']
            boxes=resize_boxes(image_shape,origin_shape)(boxes)
            result[i]['boxes']=boxes
        return result



class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size,(list,tuple)):  #这一步感觉没啥用
            min_size=(min_size,)
        self.min_size=min_size
        self.max_size=max_size
        self.image_mean = torch.as_tensor(image_mean)  # 指定图像在标准化处理中的均值,传入一个列表
        self.image_std = torch.as_tensor(image_std)  # 指定图像在标准化处理中的方差
    def forward(self,images,targets):  #images是列表不是batch
        original_sizes=[]
        for i,img in enumerate(images):
            original_sizes.append((img.shape[-2],img.shape[-1]))

            img=normalize(self.image_mean,self.image_std)(img)
            origin_size=img.shape[-2:]
            img=resize_image(self.min_size,self.max_size)(img)
            new_size=img.shape[-2:]
            images[i]=img
            if targets is not None:
                target_=targets[i]
                target_=resize_boxes(origin_size,new_size)(target_)
                targets[i]=target_
            else:
                target_=None
        image_sizes=[img.shape[-2:] for img in images]
        max_size=find_max()([list(img.shape) for img in images])
        batch_shape=[len(images)]+max_size
        batch_images=to_batch_images(batch_shape)(images)

        image_sizes_list=[]
        for image_size in image_sizes:
            image_sizes_list.append((image_size[0],image_size[1]))

        image_list=Imagelist(batch_images,image_sizes_list,original_sizes)
        return image_list,targets











