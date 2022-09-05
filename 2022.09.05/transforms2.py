import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import random
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self,image,target):
        for t in self.transforms:
            image,target=t(image,target)
        return image,target

class ToTensor(object):
    def __call__(self, image, target):
        image=F.to_tensor(image)
        return image,target

class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target

def _resize_image(image,self_min_size,self_max_size):
    img_shape=torch.tensor(image.shape[-2:])
    min_size=torch.min(img_shape)
    max_size=torch.max(img_shape)
    ratio=self_min_size/min_size
    if ratio*max_size>self_max_size:
        ratio=self_max_size/max_size
    image=F.interpolate(image[None],scale_factor=ratio)[0]
    return image


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self,min_size,max_size,image_mean,image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size  # 指定图像的最小边长范围
        self.max_size = max_size  # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std  # 指定图像在标准化处理中的方差
    def normalize(self,image):
        dtype,device=image.dtype,image.device
        mean=torch.as_tensor(self.image_mean,dtype=dtype,device=device)
        std=torch.as_tensor(self.image_std,dtype=dtype,device=device)
        image=(image-mean[:,None,None])/std[:,None,None]
        return image

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]
    def resize(self,image,target):
        h, w = image.shape[-2:]
        image=_resize_image(image,self.min_size,self.max_size)
        if target is not None:
            return image,target
        bbox=target['boxes']
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target
    def max_by_axis(self,the_list):
        maxes=the_list[0]
        for list in the_list[1:]:
            for index,item in enumerate(list):
                maxes[index]=torch.max(maxes[index],item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        max_size=self.max_by_axis([list(img.shape) for img in images])
        stride=float(size_divisible)
        max_size[1]=int(math.ceil(max_size[1]/stride)*stride)
        max_size[2]=int(math.ceil(max_size[2]/stride)*stride)
        batch_shape=[len(images)]+max_size
        batch_imgs=images[0].new_full(batch_shape,0)
        for img,pad_img in zip(images,batch_imgs):
            pad_img[:img.shape[0],:img.shape[1],:img.shape[2]].copy_(img)
        return batch_imgs

    def postprocess(self,result,image_shapes,original_image_shapes):
        if self.training:
            return result
        for i,(pred,img_shape,original_shape) in enumerate(zip(result,image_shapes,original_image_shapes)):
           boxes=pred['boxes']
           boxes=resize_boxes(boxes,img_shape,original_shape)
           result[i]['boxes']=boxes
        return result

    def forward(self,images,targets):
        for i in len(images):
            image=images[i]
            target=targets[i] if targets is not None else None
            image=self.normalize(image)

            image,target=self.resize(image,target)
            images[i]=image
            if targets is not None:
                targets[i]=target

        images_size=[image.shape[-2:] for image in images]
        images=self.batch_images(images)

        images_size_list=[]
        for image_size in images_size:
            images_size_list.append((image_size[0],image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets




class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)






def resize_boxes(boxes,original_size,new_size):
    ratios=[
        torch.tensor(s,dtype=boxes.dtype,device=boxes.device)/torch.tensor(s_orig,dtype=boxes.dtype,device=boxes.device) for s,s_orig in zip(new_size,original_size)
    ]
    ratios_height,ratios_width=ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin=xmin*ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


