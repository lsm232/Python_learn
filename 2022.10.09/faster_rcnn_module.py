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
from torchvision.ops import MultiScaleRoIAlign 


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

def generate_anchors(size,aspect_ratio,device,dtype=torch.float32):
    size=torch.as_tensor(size,dtype=dtype,device=device)
    aspect_ratio=torch.as_tensor(aspect_ratio,dtype=dtype,device=device)
    ws=torch.sqrt(1/aspect_ratio)
    hs=1/ws
    ws=size*ws
    hs=size*hs
    base_anchors=torch.stack([-ws,-hs,ws,hs],dim=1)/2
    return base_anchors.round()

def set_cell_anchors(sizes,aspect_ratios,device,dtype):
    cell_anchors=[generate_anchors(size,aspect_ratio,device,dtype) for size,aspect_ratio in zip(sizes,aspect_ratios)]
    return cell_anchors

def grid_anchors(base_anchors,grid_sizes,strides):
    #将每个特征图的每个点上生成的base anchors，复制到原图上
    anchors=[]
    for base_anchors_per_level,grid_size_per_level,stride_per_level in zip(base_anchors,grid_sizes,strides):
        shifts_x=torch.arange(0,grid_size_per_level[0])*strides[0]
        shifts_y=torch.arange(0,grid_size_per_level[1])*strides[1]
        shifts_y,shifts_x=torch.meshgrid([shifts_y,shifts_x])
        shifts_y=shifts_y.reshape(-1)
        shifts_x=shifts_x.reshape(-1)
        shifts=torch.stack([shifts_x,shifts_y,shifts_x,shifts_y],dim=1)
        shifts_anchors=base_anchors[None,:,:]+shifts[:,None,:]
        anchors.append(shifts_anchors.reshape(-1,4))
    return anchors

def cached_grid_anchors(base_anchors,cache,image_list,features):
    pad_image_size=image_list.tensor.shape[-2:]
    grid_sizes=[feature.shape[-2:] for feature in features]
    strides=[[pad_image_size[0]/grid_size[0],pad_image_size[1]/grid_size[1]] for grid_size in grid_sizes]

    key=str(grid_sizes)+str(strides)
    if key in cache:
        return cache[key]
    anchors=grid_anchors(base_anchors,grid_sizes,strides)
    cache[key]=anchors
    return anchors

class AnchorsGenerator(nn.Module):
    def __init__(self,anchor_sizes=((32,),(64,),(128,),(256,),(512,)),aspect_ratios=((0,5,1.0,2.0))):
        super(AnchorsGenerator, self).__init__()
        self.sizes=anchor_sizes
        self.aspect_ratios=len(anchor_sizes)*aspect_ratios
    def forward(self,image_list,features):
        device=features.device
        dtype=features.dtype
        base_anchors_over_features=set_cell_anchors(self.sizes,self.aspect_ratios,device,dtype)
        shifts_anchors_over_features=cached_grid_anchors(base_anchors_over_features,{},image_list,features)

        anchors=[]
        for i in range(len(image_list.tensors)):
            anchors_per_image=[]
            for anchors_per_image_shifts in shifts_anchors_over_features:
                anchors_per_image.append(anchors_per_image_shifts)
            anchors.append(anchors_per_image)

        anchors=[torch.cat(anchors_) for anchors_ in anchors]
        return anchors

def permete_and_flatten(layer,N,A,C,H,W):
    layer=layer.view(N,A,C,H,W)
    layer=layer.permute(0,3,4,1,2)
    layer=layer.reshape(N,-1,C)
    return layer

def concat_box_prediction_layers(box_cls,box_regression):
    box_cls_flatten=[]
    box_regression_flatten=[]
    for box_cls_per_level,box_regression_per_level in zip(box_cls,box_regression):
        N,AxC,H,W=box_cls_per_level.shape
        Ax4=box_regression_per_level.shape[1]
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        box_cls_per_level=permete_and_flatten(box_cls_per_level,N,A,C,H,W)
        box_cls_flatten.append(box_cls_per_level) #N,-1,C
        box_regression_per_level=permete_and_flatten(box_regression_per_level,N,A,4,H,W)
        box_regression_flatten.append(box_regression_per_level)

    box_cls_flatten=torch.cat(box_cls_flatten,dim=1).flatten(0,-2)
    box_regression_flatten=torch.cat(box_regression_flatten,dim=1).flatten(0,-2)
    return box_cls,box_regression

def decode_single(rel_codes,boxes):
    dx=rel_codes[:,0::4]
    dy=rel_codes[:,1::4]
    dw=rel_codes[:,2::4]
    dh=rel_codes[:,3::4]

    width=boxes[:,2]-boxes[:,0]
    height=boxes[:,3]-boxes[:,1]
    ctr_x=boxes[:,0]+width/2
    ctr_y=boxes[:,1]+height/2

    pred_ctr_x=dx*width[:,None]+ctr_x[:,None]
    pred_ctr_y=dy*height[:,None]+ctr_y[:,None]
    pred_width=torch.exp(dw)*width[:,None]
    pred_height=torch.exp(dh)*height[:,None]

    pred_xmin=pred_ctr_x-0.5*width
    pred_xmax=pred_ctr_x+0.5*width
    pred_ymin=pred_ctr_y-0.5*height
    pred_ymax=pred_ctr_y+0.5*height

    pred_boxes=torch.stack([pred_xmin,pred_ymin,pred_xmax,pred_ymax],dim=2).flatten(1)
    return pred_boxes

def decode(rel_codes,boxes):
    concat_boxes=torch.cat(boxes,dim=0)
    box_num=len(concat_boxes)
    pred_boxes=decode_single(rel_codes,concat_boxes)
    pred_boxes=pred_boxes.reshape(box_num,-1,4)
    return pred_boxes

def _get_top_n_idx(objectness,num_anchors_per_level,used_anchors_num):
    #(batch,num_anchors_per_image),[num1,...,num5]
    r=[]
    offset=0.
    for ob in objectness.split(num_anchors_per_level,1):
        num_anchors=ob.shape[1]
        nums=min(num_anchors,used_anchors_num)
        _,top_n_idx=ob.topk(nums,dim=1)
        r.append(top_n_idx+offset)
        offset+=nums
    return torch.cat(r,dim=1)

def clip_boxes_to_image(boxes,resized_shape):
    dim=boxes.dim()
    boxes_x=boxes[...,0::2]
    boxes_y=boxes[...,1::2]
    height,width=resized_shape

    boxes_x=boxes_x.clamp(min=0,max=width)
    boxes_y=boxes_y.clamp(min=0,max=height)

    clipped_boxes=torch.stack([boxes_x,boxes_y],dim=dim)
    return clipped_boxes.reshape(boxes.shape)

def remove_small_boxes(boxes,min_size):
    ws,hs=boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]
    keep=torch.logical_and(torch.ge(ws,min_size),torch.ge(hs,min_size))
    keep=torch.where(keep)[0]
    return keep

def batched_nms(boxes,scores,level,iou_threshold):
    max_coordinate=boxes.max()
    offsets=level*max_coordinate
    boxes=boxes+offsets[:,None]
    keep=torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return keep

def filter_proposals(proposals,objectness,resized_shapes,num_anchors_per_level):
    len_images=len(proposals)
    objectness=objectness.reshape(len_images,-1)

    levels=[torch.full(size=(n,),fill_value=i) for i,n in enumerate(num_anchors_per_level)]
    levels=torch.cat(levels).reshape(1,-1).expand_as(objectness)

    idx=_get_top_n_idx(objectness,num_anchors_per_level,500)

    image_range=torch.arange(len_images)
    batch_idx=image_range[:,None]

    proposals=proposals[batch_idx,idx]
    objectness=objectness[batch_idx,idx]
    levels=levels[batch_idx,idx]

    objectness_prob=torch.sigmoid(objectness)
    final_scores=[]
    final_boxes=[]
    for boxes,scores,lvl,resized_shape in zip(proposals,objectness_prob,levels,resized_shapes):
        boxes=clip_boxes_to_image(boxes,resized_shape)

        keep=remove_small_boxes(boxes,500)
        boxes,scores,lvl=boxes[keep],scores[keep],lvl[keep]

        keep=torch.where(torch.ge(scores,0.7))[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        keep=batched_nms(boxes,scores,lvl,0.7)
        pos_nms_top=200
        keep=keep[:pos_nms_top]
        boxes,scores=boxes[keep],scores[keep]
        final_boxes.append(boxes)
        final_scores.append(scores)
    return final_boxes,final_scores

def box_area(boxes):
    return (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

def box_iou(boxes1,boxes2):
    area1=box_area(boxes1)
    area2=box_area(boxes2)
    lt=torch.max(boxes1[:,None,:2],boxes2[None,:,:2])
    rb=torch.max(boxes1[:,None,2:],boxes2[None,:,2:])

    wh=(rb-lt).clamp(min=0)
    inter=wh[:,:,0]*wh[:,:,1]
    iou=inter/(area1[:,None]+area2-inter)
    return iou

def set_low_quality_matches_(matches,all_matches,match_quality_matrix):
    highest_quality_foreach_gt,_=match_quality_matrix.max(dim=1)
    gt_pred_pairs_of_highest_quality=torch.where(torch.eq(match_quality_matrix,highest_quality_foreach_gt[:,None]))
    pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
    matches[pre_inds_to_update] = all_matches[pre_inds_to_update]

class Matcher(object):
    def __init__(self,high_threshold,low_threshold,allow_low_quality_matches):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold  # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches
    def __call__(self,match_quality_matrix):
        matched_vals,matches=match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches=matches.clone()
        else:
            all_matches=None
        below_low_threshold=matched_vals<self.low_threshold
        between_threshold=(matched_vals>=self.low_threshold) & (matched_vals<self.high_threshold)

        matches[below_low_threshold]=self.BELOW_LOW_THRESHOLD
        matches[between_threshold]=self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            set_low_quality_matches_(matches,all_matches,match_quality_matrix)

        return matches

def assign_targets_to_anchors(anchors,targets):
    labels=[]
    matched_gt_boxes=[]
    for anchors_per_image,targets_per_image in zip(anchors,targets):
        gt_boxes=targets["boxes"]
        match_quality_matrix=box_iou(gt_boxes,anchors_per_image)
        matched_idxs=Matcher()(match_quality_matrix)

        matched_gt_boxes_per_image=gt_boxes[matched_idxs.clamp(min=0)]
        labels_per_image=matched_idxs>=0
        bg_indices=matched_idxs==Matcher().BELOW_LOW_THRESHOLD
        labels_per_image[bg_indices]=0.0

        inds_to_discard=matched_idxs==Matcher().BETWEEN_THRESHOLDS
        labels_per_image[inds_to_discard]=-1.0

    labels.append(labels_per_image)
    matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels,matched_gt_boxes

def encode_boxes(gt_boxes,anchors):
    gt_xmin=gt_boxes[:,0::4]
    gt_xmax=gt_boxes[:,1::4]
    gt_ymin=gt_boxes[:,2::4]
    gt_ymax=gt_boxes[:,3::4]

    anchors_xmin = anchors[:, 0::4]
    anchors_xmax = anchors[:, 1::4]
    anchors_ymin = anchors[:, 2::4]
    anchors_ymax = anchors[:, 3::4]

    gt_widths=gt_xmax-gt_xmin
    gt_heights=gt_ymax-gt_ymin
    anchors_widths = anchors_xmax - anchors_xmin
    anchors_heights = anchors_ymax - anchors_ymin

    gt_ctr_x=gt_xmin+0.5*gt_widths
    gt_ctr_y=gt_ymin+0.5*gt_heights
    anchors_ctr_x = anchors_xmin + 0.5 * anchors_widths
    anchors_ctr_y = anchors_ymin + 0.5 * anchors_heights

    targets_dx=(gt_ctr_x-anchors_ctr_x)/anchors_widths
    targets_dy=(gt_ctr_y-anchors_ctr_y)/anchors_heights
    targets_dw=torch.log(gt_widths/anchors_widths)
    targets_dh=torch.log(gt_heights/anchors_heights)

    targets=torch.cat([targets_dx,targets_dy,targets_dw,targets_dh],dim=1)
    return targets

def encode_single(gt_boxes,anchors):
    targets=encode_boxes(gt_boxes,anchors)
    return targets

def encode(gt_boxes,anchors):
    boxes_per_image=[len(anchors_) for anchors_ in anchors]
    gt_boxes=torch.cat(gt_boxes,dim=0)
    anchors=torch.cat(anchors,dim=0)
    targets=encode_single(gt_boxes,anchors)
    return targets.split(boxes_per_image,0)

class BalancedPositiveNegativeSampler(object):
    def __init__(self,batch_size_per_image,positive_fraction):
        self.batch_size_per_image=batch_size_per_image
        self.positive_fraction=positive_fraction

    def __call__(self,labels):
        batch_size_per_image=self.batch_size_per_image
        fraction=self.positive_fraction
        pos_idx=[]
        neg_idx=[]
        for matched_idxs_per_image in labels:
            positive=torch.where(torch.ge(matched_idxs_per_image,1))[0]
            negative=torch.where(torch.eq(matched_idxs_per_image,0))[0]
            num_pos=int(batch_size_per_image*fraction)
            num_pos=min(positive.numel(),num_pos)
            num_neg=batch_size_per_image-num_pos
            num_neg=min(negative.numel(),num_neg)
            perm1=torch.randperm(positive.numel())[:num_pos]
            perm2=torch.randperm(negative.numel())[:num_neg]
            pos_idx_per_image=positive[perm1]
            neg_idx_per_image=negative[perm2]
            pos_idx_per_image_mask=torch.zeros_like(matched_idxs_per_image)
            neg_idx_per_image_mask=torch.zeros_like(matched_idxs_per_image)
            pos_idx_per_image_mask[pos_idx_per_image]=1
            neg_idx_per_image_mask[neg_idx_per_image]=1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx,neg_idx

def smooth_l1_loss(input,target,beta,size_average):
    n=torch.abs(input-target)
    cond=torch.lt(n,beta)
    loss=torch.where(cond,0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def compute_loss(objectness,pred_box_deltas,labels,regression_targets):
    sampled_pos_inds,sampled_neg_inds=BalancedPositiveNegativeSampler(500,0.5)(labels)
    sampled_pos_inds=torch.where(torch.cat(sampled_pos_inds,dim=0))[0]
    sampled_neg_inds=torch.where(torch.cat(sampled_neg_inds,dim=0))[0]
    sampled_inds=torch.cat([sampled_pos_inds,sampled_neg_inds],dim=0)
    objectness=objectness.flatten()
    labels=torch.cat(labels,dim=0)
    regression_targets=torch.cat(regression_targets,dim=0)
    box_loss=smooth_l1_loss(pred_box_deltas,regression_targets,1/9,False)/(sampled_inds.numel())
    objectness_loss=F.binary_cross_entropy_with_logits(objectness[sampled_inds],labels[sampled_inds])
    return box_loss,objectness_loss

class RegionProposalNetwork(nn.Module):
    def __init__(self,anchors_generator,rpn_head,fg_iou_thresh,bg_iou_thresh,batch_size_per_image,positive_fraction,pre_nms_top_n,pos_nms_top_n,nms_thresh,score_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchors_generator=anchors_generator
        self.rpn_head=rpn_head
        self.fg_iou_thresh=fg_iou_thresh
        self.bg_iou_thresh=bg_iou_thresh
        self.batch_size_per_image=batch_size_per_image
        self.positive_fraction=positive_fraction
        self.pre_nms_top_n=pre_nms_top_n
        self.pos_nms_top_n=pos_nms_top_n
        self.nms_thresh=nms_thresh
        self.score_thresh=score_thresh
    def forward(self,images_list,features,targets):
        features=list(features.values())
        objectness,pred_bbox_deltas=self.rpn_head(features)
        anchors=self.anchors_generator(images_list,features)
        num_images=len(anchors)
        num_anchors_per_level_shape=[o[0].shape for o in objectness]
        num_anchors_per_level=[s[0]*s[1]*s[2] for s in num_anchors_per_level_shape]

        objectness,pred_bbox_deltas=concat_box_prediction_layers(objectness,pred_bbox_deltas)
        proposals=decode(pred_bbox_deltas,anchors)
        proposals=proposals.view(num_images,-1,4)
        boxes,scores=filter_proposals(proposals,objectness,images_list.shapes,num_anchors_per_level)

        losses={}
        if self.training:
            labels,matched_gt_boxes=assign_targets_to_anchors(anchors,targets)
            regression_targets=encode(matched_gt_boxes,anchors)
            loss_objectness,loss_rpn_box_reg=compute_loss(objectness,pred_bbox_deltas,labels,regression_targets)
            losses={
                'loss_objectness':loss_objectness,
                'loss_rpn_box_reg':loss_rpn_box_reg
            }
        return boxes,losses

def check_targets(targets):
    assert targets is not None
    assert all(["boxes" in t for t in targets])
    assert all(["labels" in t for t in targets])

def add_gt_proposals(gt_boxes,proposals):
    return [torch.cat(gt_boxes,proposal) for gt_box,proposal in zip(gt_boxes,proposals)]

def select_training_sample(targets,proposals):
    check_targets(targets)
    dtype=proposals[0].dtype
    device=proposals[0].device
    gt_boxes=[gt["boxes"] for gt in targets]
    gt_labels=[gt["labels"] for gt in targets]
    proposals=add_gt_proposals(targets["boxes"],proposals)
    matched_idx,labels=assign_targets_to_anchors(proposals,targets)
    sampled_inds=BalancedPositiveNegativeSampler(500,0.5)(labels)

    matched_gt_boxes=[]
    num_images=len(proposals)

    for img_id in range(num_images):
        img_sampled_inds=sampled_inds[img_id]
        proposals[img_id]=proposals[img_id][img_sampled_inds]
        labels[img_id]=labels[img_id][img_sampled_inds]
        matched_idx[img_id]=matched_idx[img_id][img_sampled_inds]
        
        gt_boxes_in_image=gt_boxes[img_id]
        matched_gt_boxes.append(gt_boxes_in_image[matched_idx[img_id]])
    regression_targets=encode(matched_gt_boxes,proposals)
    return proposals,labels,regression_targets

box_roi_pool=MultiScaleRoIAlign(
    featmap_names=['0','1','2','3'],
    output_size=[7,7],
    sampling_ratio=2
)

class TwoMLPHead(nn.Module):
    def __init__(self,in_channels,representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6=nn.Linear(in_channels,representation_size)
        self.fc7=nn.Linear(representation_size,representation_size)
    def forward(self,x):
        x=x.flatten(start_dim=1)
        x=F.relu(self.fc6(x))
        x=F.relu(self.fc7(x))
        return x

class FastRCNNPredictor(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score=nn.Linear(in_channels,num_classes)
        self.bbox_pred=nn.Linear(in_channels,num_classes*4)
    def forward(self,x):
        x=x.flatten(start_dim=1)
        scores=self.cls_score(x)
        bbox_deltas=self.bbox_pred(x)
        return scores,bbox_deltas

def fastrcnn_loss(class_logits,box_regression,labels,regression_targets):
    labels=torch.cat(labels,dim=0)
    regression_targets=torch.cat(regression_targets,dim=0)
    classification_loss=F.cross_entropy(class_logits,labels)

    sampled_pos_inds_subset=torch.where(torch.ge(labels,0))[0]
    labels_pos=labels[sampled_pos_inds_subset]
    N,num_classes=class_logits.shape
    box_regression=box_regression.reshape(N,-1,4)

    box_loss=smooth_l1_loss(
        box_regression[sampled_pos_inds_subset,labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1/9,
        size_average=False,
    )/labels.numel()

    return classification_loss,box_loss

def postprocess_detections(class_logits,box_regression,proposals,image_shapes):
    score_thresh=0.5
    nms_thresh=0.5
    detection_per_img=100

    num_classes=class_logits.shape[-1]
    boxes_per_image=[boxes_in_image.shape[0]  for boxes_in_image in proposals]
    pred_boxes=decode(box_regression,proposals)

    pred_scores=F.softmax(class_logits,-1)
    pred_boxes_list=pred_boxes.split(boxes_per_image,0)
    pred_scores_list=pred_scores.split(boxes_per_image,0)

    all_boxes=[]
    all_scores=[]
    all_labels=[]
    for boxes,scores,image_shape in zip(pred_boxes_list,pred_scores_list,image_shapes):
        boxes=clip_boxes_to_image(boxes,image_shape)
        labels=torch.arange(num_classes)
        labels=labels.view(1,-1).expand_as(scores)

        boxes=boxes[:,1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        inds=torch.where(torch.gt(scores,score_thresh))[0]
        boxes,scores,labels=boxes[inds],scores[inds],labels[inds]
        keep=remove_small_boxes(boxes,min_size=1)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        keep = batched_nms(boxes, scores, labels, nms_thresh)

        # keep only topk scoring predictions
        # 获取scores排在前topk个预测目标
        keep = keep[:detection_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels

class ROIheads(nn.Module):
    def __init__(self,box_roi_pool,box_head,box_predictor,fg_iou_thresh,bg_iou_thresh,
                 batch_size_per_image,positive_fraction,score_thresh,nms_thresh,detection_per_image
                 ):
        super(ROIheads, self).__init__()
        self.box_roi_pool = box_roi_pool  # Multi-scale RoIAlign pooling
        self.box_head = box_head  # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh  # default: 0.5
        self.detection_per_img = detection_per_image  # default: 100

    def forward(self,features,proposals,image_shapes,targets):
        if self.training:
            proposals,labels,regression_targets=select_training_sample(targets,proposals)
        else:
            labels=None
            regression_targets=None
        box_features=box_roi_pool(features,proposals,image_shapes)
        box_features=TwoMLPHead(backbone.out_channels*resolution**2,representation_size)(box_features)
        class_logits,box_regression=FastRCNNPredictor(1024,num_classes=5)(box_features)
        result=[]
        losses={}
        if self.training:
            loss_classifier,loss_box_reg=fastrcnn_loss(class_logits,box_regression,labels,regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes,scores,labels=postprocess_detections(class_logits,box_regression,proposals,image_shapes)
            num_images=len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes":boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result,losses





















































