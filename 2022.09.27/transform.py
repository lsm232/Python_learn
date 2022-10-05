import torch
import torch.nn as nn




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




    def forward(self,images,targets):
        for i in range(len(images)):
            image=images[i]
            target_index=targets[i]

            image=self.normalize(image)
            image,target_index=self.resize(image,target_index)


