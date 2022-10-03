import torch.nn as nn
from . import box_ops

class ROIHeads(nn.Module):
    def __init__(self,box_roi_pool,box_head,box_predictor,fg_iou_thresh,bg_thresh_iou,batch_size_per_image,positive_fraction,
                 bbox_reg_weights,score_thresh,nms_thresh,detection_per_image
                 ):
        super(ROIHeads, self).__init__()
        self.box_similarity=box_ops.box_iou
        self.proposal_matcher=det_utils.Matcher(fg_iou_thresh,bg_thresh_iou,allow_low_quality_matches=False)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool  # Multi-scale RoIAlign pooling
        self.box_head = box_head  # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh  # default: 0.5
        self.detection_per_img = detection_per_img