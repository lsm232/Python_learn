import torch
import torch.nn as nn

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


    def forward(self,image_list,feature_maps):



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



