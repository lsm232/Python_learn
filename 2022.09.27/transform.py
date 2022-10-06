import torch
import math
import torch.nn as nn
from typing import List



class GeneralizedRCNNTransform(nn.Module):
    def __init__(self,min_size,max_size,image_mean,image_std):
        super(GeneralizedRCNNTransform, self).__init__()

        self.min_size=min_size
        self.max_size = max_size  # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std

    def normalize(self,image):
        mean=torch.as_tensor(self.image_mean)
        std=torch.as_tensor(self.image_std)
        return (image-mean[:,None,None])/std[:,None,None]

    def _resize_image(self,image,self_min_size,self_max_size):
        im_shape=torch.tensor(image.shape[-2:])
        min_size=float(torch.min(im_shape))
        max_size=float(torch.max(im_shape))
        scale_factor=self_min_size/min_size

        if scale_factor*max_size>self_max_size:
            scale_factor=self_max_size/max_size

        image=torch.nn.functional.interpolate(image,scale_factor=scale_factor)
        return image

    def resize(self,image,target):
        h,w=image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size =float(self.min_size[-1])

        image= self._resize_image(image, size, float(self.max_size))
        if target is None:
            return image,target

        bbox=target["boxes"]
        bbox=resize_boxes(bbox,[h,w],image.shape[-2:])
        target["boxes"]=bbox
        return image,target

    def resize_boxes(self,boxes,original_shapes,resized_shapes):
        ratios=[torch.tensor(s)/torch.tensor(o) for s,o in zip(original_shapes,resized_shapes)]
        ratio_height,ratio_width=ratios
        xmin,ymin,xmax,ymax=boxes.unbind(1)
        xmin=xmin*ratio_width
        xmax=xmax*ratio_width
        ymin=ymin*ratio_height
        ymax=ymax*ratio_height
        return torch.stack([xmin,ymin,xmax,ymax],dim=1)


    def max_by_axis(self,resized_shapes):
        # type:(List[List[int]]) -> List[int]
        maxes=resized_shapes[0]
        for son_list in resized_shapes[1:]:
            for index,item in son_list:
                maxes[index]=max(maxes[index],item)
        return maxes


    def batch_images(self,resized_images,size_divisible=32):
        max_size=self.max_by_axis([list(img.shape) for img in resized_images])
        max_size[0]=int(math.ceil(float(max_size[0])/size_divisible)*size_divisible)
        max_size[1]=int(math.ceil(float(max_size[1])/size_divisible)*size_divisible)
        batch_shape=list(len(resized_images))+max_size

        batch_images=resized_images[0].new_full(batch_shape,0)

        for img,pad_img in zip(resized_images,batch_images):
            pad_img[:img.shape[0],:img.shape[1],:img.shape[2]].copy_(img)

        return batch_images



    def forward(self,images,targets):
        for i in range(len(images)):
            image=images[i]
            target_index=targets[i]

            image=self.normalize(image)
            image,target_index=self.resize(image,target_index)
            images[i]=image
            if target_index is not None:
                targets[i]=targets

        image_sizes=[img.shape[-2:] for img in images]
        images=self.batch_images(images)

        image_list=ImageList(images, image_sizes_list)
        return image_list, targets





