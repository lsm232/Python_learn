import torch
from typing import List,Tuple,Dict
from torch import Tensor
import torch.nn as nn
from .det_utils import *

class AnchorGenerator(nn.Module):
    def __init__(self,sizes,aspect_ratios):
        super(AnchorGenerator, self).__init__()
        self.sizes=sizes   #((64,),(128,),(256,))
        self.aspect_ratios=aspect_ratios   #(0.5,1,2)
        self._cache={}

    def gird_anchors(self,grid_sizes,strides):
        #type: (List[List[int]],List[List[Tensor]]) -> List[Tensor]
        shifts_=[]
        for grid_sizes,strides in zip(grid_sizes,strides):
            f_h,f_w=grid_sizes
            s_h,s_w=strides
            hs=torch.arange(0,f_h)*s_h   #torch.range 返回float且包括end，arange返回int,不包括end
            ws=torch.arange(0,f_w)*s_w
            shift_y,shift_x=torch.meshgrid(hs,ws)
            shift_y=shift_y.reshape(-1)
            shift_x=shift_x.reshape(-1)
            shifts=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1)
            shifts_.append(shifts)
        return shifts_   #(n,4)

    def generate_anchors(self,sizes,aspect_raios):
        sizes=torch.as_tensor(sizes)
        aspect_raios=torch.as_tensor(aspect_raios)
        h_ratios=torch.sqrt(aspect_raios)
        w_ratios=1/h_ratios
        ws=(w_ratios[:,None]*sizes[None,:]).view(-1)
        hs=(h_ratios[:,None]*sizes[None,:]).view(-1)
        base_anchors=torch.stack([-ws,-hs,ws,hs],dim=1)/2
        return base_anchors.round()  #(3,4)


    def set_cell_anchors(self):
        cell_anchors=[
            self.generate_anchors(sizes,aspect_ratios)
            for sizes,aspect_ratios in zip(self.sizes,self.aspect_ratios)
        ]
        self.cell_anchors=torch.cat(cell_anchors,dim=0)

    def cached_gird_anchors(self,gird_sizes,strides):
        # type: (List[List[int]],List[List[Tensor]]) -> List[Tensor]
        key=str(gird_sizes)+str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors=self.grid_anchors(gird_sizes,strides)
        self._cache[key]=anchors
        return anchors


    def forward(self,imagelist,feature_maps):
        #求出feature map与原图像的比，用于移动anchors
        grid_sizes=list([feature_map.shape[-2:] for feature_map in feature_maps])  #feature_map.shape[-2:]=torch.Size([200,500]),torch.tensor(torch.Size([200,500]))=tensor([200,500])
        pad_sizes=imagelist.batch_imgs.shape[-2:]
        strides=[[torch.tensor(pad_sizes[0]//g[0]),torch.tensor(pad_sizes[1]//g[1])] for g in grid_sizes]  #torch.tensor不对数据类型进行转换，torch.Tensor是torch.FloatTensor的缩写，会将数据转换为浮点型

        #计算偏移量
        shifts=self.cached_gird_anchors(grid_sizes,strides)

        #生成anchors
        self.set_cell_anchors()

        #将偏移量与anchors相加，相当于移动anchors
        shifts_anchors=shifts.view(-1,1,4)+self.cell_anchors.view(1,-1,4)
        shifts_anchors=shifts_anchors.reshape(-1,4)

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

def concat_box_prediction_layers(box_cls,box_regression):
    box_cls_flatten=[]
    box_regression_flatten=[]
    for box_cls_per_level,box_regression_per_level in zip(box_cls,box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        # classes_num
        C = AxC // A
        box_cls_per_level=box_cls_per_level.reshape(N,-1,C,H,W)
        box_cls_per_level=box_cls_per_level.permute(0,3,4,1,2)
        box_cls_per_level=box_cls_per_level.reshape(N,-1,C)
        box_regression = box_regression.reshape(N, -1, 4, H, W)
        box_regression = box_regression.permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)
        box_cls_flatten.append(box_cls_per_level)
        box_regression_flatten.append(box_regression_per_level)
    box_cls=torch.cat(box_cls_flatten,dim=1).flatten(0,-2)
    box_regression = torch.cat(box_regression_flatten, dim=1).reshape(-1, 4)
    return box_cls, box_regression





class RegionProposoalNetwork(nn.Module):
    def __init__(self,anchorsGenerator,weights,nms_thresh,fg_iou_thresh,bg_iou_thresh):
        super(RegionProposoalNetwork, self).__init__()

        self.anchorsGenerator=anchorsGenerator
        self.box_coder=BoxCoder(weights)
        self.nms_thresh=nms_thresh
        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']
    def pre_nms_top_n(self):
        if self.istraining:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def get_top_n_idx(self,objectness,num_anchors_per_level):
        r=[]
        offset=0
        for ob in objectness.split(num_anchors_per_level,dim=1):
            num_anchors=ob.shape[1]
            pre_nms_top_n=min(self.pre_nms_top_n(),num_anchors)
            _,top_n_idx=ob.topk(pre_nms_top_n,dim=1)
            r.append(top_n_idx+offset)
            offset+=num_anchors
        return torch.cat(r,dim=1)

    def clip_boxes_to_image(self, boxes, shape):
        dim=boxes.dim()
        boxes_x=boxes[:,0::2]
        boxes_y = boxes[:, 1::2]
        height,width=shape

        boxes_x=boxes_x.clamp(min=0,max=width)
        boxes_y=boxes_y.clamp(min=0,max=height)
        clipped_boxes=torch.stack((boxes_x,boxes_y),dim)
        return clipped_boxes.reshape(boxes.shape)

    def remove_small_boxes(self,boxes,min_size):
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep=torch.logical_and(torch.ge(ws,min_size),torch.ge(hs,min_size))

        keep = torch.where(keep)[0]
        return keep

    def batched_nms(self,boxes,scores,lvl,nms_thresh):
        max_coordinate=boxes.max()
        offsets=lvl*(max_coordinate+1)
        boxes_for_nms=boxes+offsets[:,None]
        keep=torch.ops.torchvision.nms(boxes,scores,nms_thresh)
        return keep





    def filter_proposals(self,proposals,objectness,image_shapes,num_anchors_per_level):
        num_imgs=proposals.shape[0]
        objectness=objectness.detach()

        objectness=objectness.reshape(num_imgs,-1)

        levels=[torch.full((num_anchors,),idx,dtype=torch.int32) for idx,num_anchors in enumerate(num_anchors_per_level)]
        levels=torch.cat(levels,dim=0)
        levels=levels.reshape(1,-1).expand_as(objectness)

        top_n_idx=self.get_top_n_idx(objectness,num_anchors_per_level)
        image_range=torch.arange[num_imgs]
        batch_idx=image_range[:,None]

        objectness=objectness[batch_idx,top_n_idx]
        levels=levels[batch_idx,top_n_idx]
        proposals=proposals[batch_idx,top_n_idx]

        objectness_prob=torch.sigmoid(objectness)
        final_boxes=[]
        final_scores=[]
        for boxes,scores,lvl,img_shape in zip(proposals,objectness,levels,image_shapes):
            boxes=self.clip_boxes_to_image(boxes,image_shapes)
            keep=self.remove_small_boxes(boxes,min_size=self.min_size)
            boxes,scores,lvl=boxes[keep],scores[keep],lvl[keep]
            keep=self.batched_nms(boxes,scores,lvl,self.num_thresh)
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def box_iou(self,gt_boxes,anchors_per_image):
        area1=(gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:, 3] - gt_boxes[:, 1])
        area2=(anchors_per_image[:,2]-anchors_per_image[:,0])*(anchors_per_image[:, 3] - anchors_per_image[:, 1])
        lt=torch.max(gt_boxes[:,None,:2],anchors_per_image[:,:2])
        rb=torch.max(gt_boxes[:,None,2:],anchors_per_image[:,2:])
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou








    def assign_targets_to_anchors(self,anchors,targets):
        labels=[]
        matched_gt_boxes=[]
        for anchors_per_image,targets_per_image in zip(anchors,targets):
            gt_boxes=targets['boxes']
            if gt_boxes.numel()==0:
                matched_gt_boxes_per_image=torch.zeros(anchors_per_image)
                labels_per_image=torch.zeros((anchors_per_image.shape[0]))
            else:
                match_quality_matrix=self.box_iou(gt_boxes,anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)









    def forward(self,imagelist,features,targets):
        features=list(features.values())
        objectness,pred_bbox_deltas=self.head(features)  #list[[b,num_anchors*1,h,w],...]   list[[b,num_anchors*4,h,w],...]
        anchors=self.anchorsGenerator(imagelist,features)  #list[(num_anchors*h1*w1+num_anchors*h2*w2...,4),()]
        num_images=len(anchors)

        num_anchors_per_level=[o.shape[1]*o.shape[2]*o.shape[3] for o in objectness]  #[int ,int ,...]
        objectness,pred_bbox_deltas=concat_box_prediction_layers(objectness,pred_bbox_deltas)  #[nums,1],[nums,4]
        proposals=self.box_coder.decode(pred_bbox_deltas.detach(),anchors)
        proposals=proposals.reshape(num_images,-1,4)

        boxes,scores=self.filter_proposals(proposals,objectness,imagelist.image_list,num_anchors_per_level)
        losses={}
        if self.training:
            labels,matched_gt_boxes=self.assign_targets_to_anchors(anchors, targets)













