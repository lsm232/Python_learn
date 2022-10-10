import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import copy
import os
from torchvision.transforms import functional as F
import random
from torch.utils.data import Dataset
from lxml import etree
import json
from PIL import Image
import bisect
from typing import Dict,List,Tuple
from torch import Tensor


device=torch.device("gpu" if torch.cuda.is_available() else "cpu")

#
class Compose(object):
    def __init__(self,funs):
        self.funs=funs
    def __call__(self,images,targets):
        for fun in self.funs:
            images,targets=fun(images,targets)
        return images,targets
class ToTensor(object):
    """将pil image转换为tensor"""
    def __call__(self, images,targets):
        #return torch.as_tensor(images),torch.as_tensor(targets)
        return F.to_tensor(images),targets
class RandomHorizontalFlip(object):
    def __init__(self,prob=0.5):
        self.prob=prob
    def __call__(self,images,targets):
        if random.random()<self.prob:
            height,width=images.shape[-2:]
            images.flip(-1)
            bbox=targets["boxes"]
            bbox[:,[0,2]]=width-bbox[:,[2,0]]
            targets["boxes"]=bbox
        return images,targets
#
data_transform={
    "train":Compose([ToTensor(),RandomHorizontalFlip(0.5)]),
    "val":Compose([ToTensor()]),
}
#
class vocDataset(Dataset):
    def __init__(self, root=r'G:\leran to play\VOCdevkit\VOC2012', transforms=data_transform['train'], txt_name="train.txt"):
        self.root = root
        self.images_path = os.path.join(self.root, "JPEGImages")
        self.annotations_path = os.path.join(self.root, "Annotations")
        self.txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)

        assert os.path.exists(self.images_path), "not found {} file".format(self.images_path)
        assert os.path.exists(self.annotations_path), "not found {} file".format(self.annotations_path)
        assert os.path.exists(self.txt_path), "not found {} file".format(self.txt_path)

        xml_list = []
        self.xml_list=[]
        with open(self.txt_path) as fid:
            xml_list.extend(os.path.join(self.annotations_path, line.strip()+".xml") for line in fid.readlines())
            #xml_list=[os.path.join(self.annotations_path, line.strip()+".xml") for line in fid.readlines()]
        for xml_path in xml_list:
            with open(xml_path) as fid:
                xml_str=fid.read()
            xml=etree.fromstring(xml_str)
            xml_dict=self.parse_xml_to_dict(xml)
            data=xml_dict["annotation"]
            if "object" not in data:
                print(f"no object in {xml_path} ,skip")
                continue
            self.xml_list.append(xml_path)

        json_file=r'./pascal_voc_classes.json'
        with open(json_file) as fid:
            self.class_dict=json.load(fid)

        self.transforms=transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        xml_file=self.xml_list[item]
        with open(xml_file) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)['annotation']

        img_path=os.path.join(self.images_path,data["filename"])
        img=Image.open(img_path)

        targets={}
        targets["labels"]=[]
        targets["is_crowd"]=[]
        targets["boxes"]=[]
        for ob in data['object']:
            targets["labels"].append((self.class_dict[ob["name"]]))
            targets["is_crowd"].append(float(ob["difficult"]))
            targets["boxes"].append([float(ob["bndbox"]["xmin"]),float(ob["bndbox"]["ymin"]),float(ob["bndbox"]["xmax"]),float(ob["bndbox"]["ymax"])])

        targets['boxes']=torch.as_tensor(targets['boxes'])
        targets['is_crowd']=torch.as_tensor(targets['is_crowd'])
        targets['labels']=torch.as_tensor(targets['labels'])


        if self.transforms is not None:
            img,targets=self.transforms(img,targets)

        return img,targets

    def parse_xml_to_dict(self,xml):
        if len(xml)==0:
            return {xml.tag:xml.text}
        result={}
        for xml_child in xml:
            child_result=self.parse_xml_to_dict(xml_child)
            if (xml_child.tag!="object"):
                result[xml_child.tag]=child_result[xml_child.tag]
            else:
                if "object" not in result:
                    result["object"]=[]
                result["object"].append(child_result["object"])
        return {xml.tag:result}

    def get_height_and_width(self,idx):
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)
        height=int(data['annotation']['size']["height"])
        width=int(data['annotation']['size']["width"])
        return height,width

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
#

train_dataset=vocDataset()
train_sampler=None

#
def _compute_aspect_ratios_custom_dataset(dataset):
    file_nums=len(dataset)
    aspect_ratios=[]
    for i in range(file_nums):
        height,width=dataset.get_height_and_width(1)
        aspect_ratios.append(height/width)
    return aspect_ratios

def _quantize(x,bins):
    bins=copy.deepcopy(bins)
    bins=sorted(bins)
    quantized=list(map(lambda y:bisect.bisect_right(bins,y),x))
    return quantized

def create_aspect_ratio_groups(dataset,k=0):
    aspect_ratios=_compute_aspect_ratios_custom_dataset(dataset)
    bins=(2**torch.linspace(-1,1,2*k+1)).tolist() if k>0 else [1.0]
    groups=_quantize(aspect_ratios,bins)
    return groups

train_data_loader=torch.utils.data.DataLoader(
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn,
)

def warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor):
    def f(x):
        if x<warmup_iters:
            return 1.0
        a=x/warmup_iters
        return warmup_factor*(1-a)+a
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=f)

scaler=torch.cuda.amp.GradScaler()

def train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=50,warmup=False,scaler=None):
    model.train()
    lr_scheduler=None
    if epoch==0 and warmup is True:
        warmup_factor=1.0/1000
        warmup_iters=1000
        lr_scheduler=warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor)

    mloss=torch.zeros(1).to(device)
    for i,(images,targets) in data_loader:
        images=list(image.to(device) for image in images)
        targets=[{k:v.to(device) for k,v in t.items()}      for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict=model(images,targets)
            losses=sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
#

def normalize(image,mean,std):
    #image的类型为tensor，mean为list, std为list
    device,dtype=image.device,image.dtype
    mean=torch.as_tensor(mean)
    std=torch.as_tensor(std)
    return (image-mean[:,None,None])/std[:,None,None]

def resize_image(image,min_size,max_size):
    h,w=image.shape[-2:]
    max_=max(h,w)
    min_=min(h,w)
    scale_factor=max_size/max_
    if scale_factor*min_size>min_size:
        scale_factor=min_size/min_
    image=torch.nn.functional.interpolate(image[None],scale_factor=scale_factor)[0]
    return image

def resize_target_boxes(boxes,original_size,new_size):
    # ratios=[n/o for n,o in zip(new_size,original_size)]
    # ratio_height,ratio_width=ratios
    # boxes[:,0]=boxes[:,0]*ratio_width
    # boxes[:,2]=boxes[:,2]*ratio_width
    # boxes[:,1]=boxes[:,1]*ratio_height
    # boxes[:,3]=boxes[:,3]*ratio_height
    ratios=[torch.as_tensor(n,dtype=torch.float32)/torch.as_tensor(o,dtype=torch.float32) for n,o in zip(new_size,original_size)]
    ratio_height,ratio_width=ratios
    xmin,xmax,ymin,ymax=boxes.unbind(1)
    xmin=xmin*ratio_width
    xmax=xmax*ratio_width
    ymin=ymin*ratio_height
    ymax=ymax*ratio_height
    boxes=torch.stack([xmin,ymin,xmax,ymax],dim=1)
    return boxes

def resize(image,target):
    h,w=image.shape[-2:]
    image=resize_image(image,200,1000)
    nh,nw=image.shape[-2:]

    if target is None:
        return image,target

    boxes=target["boxes"]
    boxes=resize_target_boxes(boxes,[h,w],[nh,nw])
    target["boxes"]=boxes
    return image,target

def max_by_axis(images):
    image=images[0]
    max_shape=[image.shape[0],image.shape[1],image.shape[2]]
    for img in images[1:]:
        for i,im in enumerate(img.shape):
            max_shape[i]=max(max_shape[i],im)
    return max_shape

def batch_image(images,size_divisible=32):
    max_size=max_by_axis(images)
    stride=float(size_divisible)

    max_size[1]=int(math.ceil(max_size[1]/stride)*stride)
    max_size[2]=int(math.ceil(max_size[2]/stride)*stride)

    batch_shape=[len(images)]+max_size
    batch_images=images[0].new_full(batch_shape,0)
    for img,batch_image in zip(images,batch_images):
        batch_image[:img.shape[0],:img.shape[1],:img.shape[2]].copy_(img)  #这里为什么改batch_image 就能改batch_images
    return batch_images

class ImageList(object):
    def __init__(self,tensors,resized_sizes):
        self.tensors=tensors
        self.image_sizes=resized_sizes

class GeneralizedRCNNTransform(nn.Module):
    def __init__(self,resized_min_size,resized_max_size,image_mean,image_std):
        self.resized_min_size=resized_min_size
        self.resized_max_size=resized_max_size
        self.image_mean=image_mean
        self.image_std=image_std
    def forward(self,images,targets):
        resized_images_size=[]
        for i in range(len(images)):
            image=images[i]
            target_index=targets[i] if targets is not None

            image=normalize(image,self.image_mean,self.image_std)
            image,target_index=resize(image,target_index)
            images[i]=image
            targets[i]=target_index

            resized_images_size.append(image.shape[-2:])

        images=batch_image(images)
        image_list=ImageList(images,resized_images_size)
        return image_list,targets

class RPNHead(nn.Module):
    def __init__(self,in_channels,num_anchors):
        super(RPNHead, self).__init__()

        self.conv=nn.Conv2d(in_channels,in_channels,3,1,1)
        self.cls_logits=nn.Conv2d(in_channels,num_anchors,1)
        self.bbox_pred=nn.Conv2d(in_channels,4*num_anchors,1)

        for layer in self.children():
            if isinstance(layer,nn.Conv2d):
                torch.nn.init.normal_(layer.weight,std=0.01)
                torch.nn.init.constant_(layer.bias,0)

    def forward(self,features):
        logits_per_level=[]
        bbox_per_level=[]
        for feature in features:
            feature=F.relu(self.conv(feature))
            logits_per_level.append(self.cls_logits(feature))
            bbox_per_level.append(self.bbox_pred(feature))
        return logits_per_level,bbox_per_level












