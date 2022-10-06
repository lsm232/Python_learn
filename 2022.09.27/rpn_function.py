import torch
import torch.nn as nn
from torch.nn import functional as F

class AnchorGenerator(nn.Module):
    def __init__(self,anchor_sizes=(128, 256, 512),aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()
        if not isinstance(anchor_sizes,(list,tuple)):
            anchor_sizes=tuple((s,) for s in anchor_sizes)
        if not isinstance(aspect_ratios,(list,tuple)):
            aspect_ratios=(aspect_ratios,)*len(anchor_sizes)

        assert len(anchor_sizes)==len(aspect_ratios)
        self.sizes=anchor_sizes
        self.aspect_ratios=aspect_ratios

        self.cell_anchors=None  #这两个是什么玩意？
        self._cache = {}

    def generate_anchors(self,size,aspect_ratio):
        size=torch.as_tensor(size)
        aspect_ratio=torch.as_tensor(aspect_ratio)
        h_ratios=torch.sqrt(aspect_ratio)
        w_ratios=1/h_ratios

        ws=(w_ratios[:,None]*size[None,:]).view(-1)
        hs=(h_ratios[:,None]*size[None,:]).view(-1)

        base_anchors=torch.stack([-ws,-hs,ws,hs],dim=1)/2
        return base_anchors.round()

    def grid_anchors(self,grid_sizes,strides):
        anchors=[]
        cell_anchors=self.cell_anchors
        assert cell_anchors is not None

        for size,stride,base_anchors in zip(grid_sizes,strides,cell_anchors):
            grid_height,grid_width=size
            stride_height,stride_width=stride

            shifts_x=torch.arange(grid_width)*stride_width
            shifts_y=torch.arange(grid_height)*stride_height

            shift_y,shift_x=torch.meshgrid(shifts_y,shifts_x)
            shift_y=shift_y.reshape(-1)
            shift_x=shift_x.reshape(-1)

            shifts=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1)
            shifts_anchors=shifts.reshape(-1,1,4)+base_anchors.rehsape(1,-1,4)
            anchors.append(shifts_anchors)  #将每一个特征图的anchors分开放[all_anchors_over_feature1,all-anchors_over_feature2,...]
        return anchors




    def set_cell_anchors(self):
        self.cell_anchors=[self.generate_anchors(size,aspect_ratio) for size,aspect_ratio in zip(self.sizes,self.aspect_ratios)]

    def cached_grid_anchors(self,grid_sizes,strides):
        key=str(grid_sizes)+str(strides)  #如果前面的batch有相同的图像尺寸，则直接返回anchors,anchors只是一个模板
        if key in self._cache:
            return self._cache[key]
        anchors=self.grid_anchors(grid_sizes,strides)

        self._cache[key] = anchors
        return anchors

    def forward(self,image_list,feature_maps):
        grid_sizes=[feature_map.shape[-2:] for feature_map in feature_maps]
        image_size=image_list.pad_images.shape[-2:]
        strides=[[torch.tensor(image_size[0]//g[0]),torch.tensor(image_size[1]//g[1])] for g in grid_sizes]
        self.set_cell_anchors()

        anchors_over_all_featrue_maps=self.cached_grid_anchors(grid_sizes,strides)





class PRNHead(nn.Module):
    def __init__(self,in_channels,num_anchors):
        super(PRNHead, self).__init__()
        self.conv=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.cls_logits=nn.Conv2d(in_channels,num_anchors,kernel_size=1)
        self.bbox_pred=nn.Conv2d(in_channels,num_anchors*4,kernel_size=1)

        for layer in self.children():
            if isinstance(layer,nn.Conv2d):
                torch.nn.init.normal_(layer.weight,std=0.01)
                torch.nn.init.constant_(layer.bias,0)

    def forward(self,x):
        logits=[]
        bbox_reg=[]

        for i,feature in enumerate(x):
            t=F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits,bbox_reg




class RegionProposalNetwork(nn.Module):
    def __init__(self,anchor_generator,head,fg_iou_thresh,bg_iou_thresh,batch_size_per_image,positive_fraction,pre_nms_top_n,post_nms_top_n,nms_thresh,score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator=anchor_generator
        self.head=head
        self.box_coder=det_utils.BoxCoder(weights=(1.0,1.0,1.0,1.0))
        self.box_similarity=box_ops.box_iou

        self.proposal_matcher=det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        self.fg_bg_sampler=det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.



    def forward(self,image_list,features,targets):
        features=list(features.values())
        objectness,pred_bbox_deltas=self.head(features)
        anchors=self.anchor_generator(image_list,features)



