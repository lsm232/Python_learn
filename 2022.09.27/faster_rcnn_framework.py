import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F


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

class FasterRCNNBase(nn.Module):
    def __init__(self,backbone,rpn,roi_heads,transform):
        super(FasterRCNNBase, self).__init__()
        self.transform=transform
        self.backbone=backbone
        self.rpn=rpn
        self.roi_heads=roi_heads
        self._has_warned = False

    def forward(self,images,targets):
        if self.training and targets is None:
            raise ValueError("no targets")
        if self.traning:
            for target in targets:
                boxes=target["boxes"]

        original_image_sizes=[]
        for img in images:
            h,w=img[:,-2:]
            original_image_sizes.append((h,w))

        images,taregts=self.transform(images,targets)





class FastRCNNPredictor(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score=nn.Linear(in_channels,num_classes)  #对每一个box进行类别预测，包括背景
        self.bbox_pred=nn.Linear(in_channels,num_classes*4)


class FasterRCNN(FasterRCNNBase):
    def __init__(self,
                 backbone,
                 num_classes=None,
                 #transform parameters
                 min_size=800,
                 max_size=1333,
                 image_mean=None,
                 image_std=None,
                 #--------------------

                 #rpn parameters
                 rpn_anchor_generator=None,
                 rpn_head=None, #rpn_generator产生的proposals(对anchors进行nms处理后得到)，经该模块输出预测类别(前景和背景)和每一个proposals和每一个gt box的偏移量
                 rpn_pre_nms_top_n_train=2000,  #特征金字塔每一层的anchors数目
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000,
                 rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7, #对rpn生成的proposals进行非极大值抑制的阈值
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, #训练时采样的proposals
                 rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0, #测试过程中，分类得分阈值
                 #-----------------------

                 #box parameters
                 box_roi_pool=None, #对特侦图进行crop和resize
                 box_head=None, #对pool处理后的特侦图进行特征提取
                 box_predictor=None, #预测类别和输出
                 #-----------------------

                 #box
                 box_score_thresh=0.05, #测试阶段，仅返回分类概率大于该阈值的box
                 box_nms_thresh=0.5, #测试阶段，
                 box_detections_per_image=100, #每张图像最大检测物体数量
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, #fast rcnn计算误差时采样的样本数
                 box_positive_fraction=0.25,
                 bbox_reg_weights=None
                 ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError("backbone should have out_channels")
        assert isinstance(rpn_anchor_generator,(AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")

        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")
        out_channels=backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes=((32,),(64,),(128,),(256,),(512,))  #这里为什么要用元组
            aspect_ratios=((0.5,1.0,2.0),)*len(anchor_sizes)
            rpn_anchor_generator=AnchorsGenerator(anchor_sizes,aspect_ratios)

        if rpn_head is None:
            rpn_head=RPNHead(out_channels,rpn_anchor_generator.num_anchors_per_location[0])  #对于rpn金字塔，每一层特征图的每个位置仅生成三个anchors，

        rpn_pre_nms_top_n=dict(training=rpn_pre_nms_top_n_train,testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n=dict(training=rpn_post_nms_top_n_train,testing=rpn_post_nms_top_n_test)

        rpn=RegionProposalNetwork(rpn_anchor_generator,rpn_head,rpn_fg_iou_thresh,rpn_bg_iou_thresh,rpn_batch_size_per_image,
                                  rpn_positive_fraction,rpn_pre_nms_top_n,rpn_post_nms_top_n,rpn_nms_thresh,score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool=MultiScaleRoIAlign(
                featmap_names=['0','1','2','3'],
                output_size=[7,7],
                sampling_ratio=2,
            )

        if box_head is None:
            resolution=box_roi_pool.output_size[0]
            representation_size=1024
            box_head=TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        if box_predictor is None:
            representation_size=1024
            box_predictor=FastRCNNPredictor(
                representation_size,
                num_classes
            )

        roi_heads=RoIHeads(
            box_roi_pool,box_head,box_predictor,
            box_fg_iou_thresh,box_bg_iou_thresh,
            box_batch_size_per_image,box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,box_nms_thresh,box_detections_per_image
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform=GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super(FasterRCNN, self).__init__()
