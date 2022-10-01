import torch
import torch.nn as nn


class FasterRCNN(nn.Module):
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
                 rpn_head=None, #rpn_generator产生的proposals(对anchors进行nms处理后得到)，经该模块输出预测类别和每一个proposals和每一个gt box的偏移量
                 rpn_pre_nms_top_n_train=2000,  #特征金字塔每一层的anchors数目
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000,
                 rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7, #对rpn生成的proposals进行非极大值抑制的阈值
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, #训练时不是采样的proposals
                 rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0, #测试过程中，分类得分阈值
                 #-----------------------

                 #box parameters
                 box_roi_pool=None, #对特侦图进行crop和resize
                 box_head=None, #对pool处理后的特侦图进行特征提取
                 box_predictor=None, #预测类别和输出
                 #-----------------------










                 ):
        super(FasterRCNN, self).__init__()