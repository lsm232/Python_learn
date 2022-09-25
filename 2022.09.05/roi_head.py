import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional,List,Dict,Tuple,Optional
from .boxes import *
from .det_utils import *
import torch.nn.functional as F



def faster_cnn_loss(class_logits,box_regression,labels,regression_targets):
    labels=torch.cat(labels,0)
    regression_targets=torch.cat(regression_targets,0)
    classification_loss=F.cross_entropy(class_logits,labels)
    sampled_pos_inds_subset=torch.where(torch.gt(labels,0))[0]
    labels_pos=labels[sampled_pos_inds_subset]
    N,num_classes=class_logits.shape
    box_regression=box_regression.reshape(N,-1,4)
    box_loss = smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss

class ROIHeads(nn.Module):
    def __init__(self,batch_size_per_image,positive_fraction,box_roi_pool):
        self.proposal_matcher=Mather
        self.fg_bg_sampler=BalancedPositiveNegativeSampler(batch_size_per_image,positive_fraction)
        self.box_roi_pool=box_roi_pool


    def add_gt_boxes(self,proposals,gt_boxes):
        proposals=[
            torch.cat([proposal,gt_box]) for (proposal,gt_box) in zip(proposals,gt_boxes)
        ]
        return proposals


    def assign_targets_to_proposals(self,proposals,gt_boxes,gt_labels):
        matched_idxs=[]
        labels=[]
        for proposals_in_image,gt_boxes_in_image,gt_labels_in_image in zip(proposals,gt_boxes,gt_labels):
            if gt_boxes_in_image.numel()==0:
                clamped_matched_idxs_in_image=torch.zeros((proposals_in_image.shape[0],),dtype=torch.int64)
                labels_in_image=torch.zeros((proposals_in_image.shape[0],),dtype=torch.int64)
            else:
                match_quality_matrix=box_iou(gt_boxes_in_image,proposals_in_image)
                matched_idxs_in_image=self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_image=matched_idxs_in_image.clamp(min=0)
                labels_in_image=gt_labels_in_image[clamped_matched_idxs_in_image]
                bg_inds=matched_idxs_in_image==self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds]=0

                ignore_inds=matched_idxs_in_image==self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds]=-1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs,labels

    def subsample(self,labels):
        sampled_pos_inds,sampled_neg_inds=self.fg_bg_sampler(labels)
        sampled_inds=[]
        for img_idx,(pos_inds_img,neg_inds_img) in enumerate(zip(sampled_pos_inds,sampled_neg_inds)):
            img_sampled_inds=torch.where(pos_inds_img|neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds











    def select_training_samples(self,
                                proposals, #type:List[Tensor]
                                targets, #type:Optional[List[Dict[str,Tensor]]]
                                ):
        gt_boxes=[t['boxes'] for t in targets]
        gt_labels=[t['labels'] for t in targets]
        proposals=self.add_gt_boxes(proposals,gt_boxes)
        matched_idxs,labels=self.assign_targets_to_proposals(proposals,gt_boxes,gt_labels)
        sampled_inds=self.subsample(labels)
        matched_gt_boxes=[]
        num_images=len(proposals)

        for img_id in range(num_images):
            img_sampled_inds=sampled_inds[img_id]
            proposals[img_id]=proposals[img_id][img_sampled_inds]
            labels[img_id]=labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4))
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
            return proposals, labels, regression_targets


    def postprocess_detections(self,class_logits,box_regression,proposals,image_shapes):
        num_classes=class_logits.shape[-1]
        boxes_per_image=[boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list=pred_boxes.split(boxes_per_image,0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes,scores,image_shape in zip(pred_boxes_list,pred_scores_list,image_shapes):
            boxes=clip_boxes_to_image(boxes,image_shape)
            labels=torch.arange(num_classes)

            labels = labels.view(1, -1).expand_as(scores)
            boxes=boxes[:,1:]
            scores=scores[:,1:]
            labels=labels[:,1:]

            boxes=boxes.reshape(-1,4)
            scores=scores.reshape(-1)
            labels=labels.reshape(-1)

            inds=torch.where(torch.gt(scores,self.score_thresh))[0]
            boxes,scores,labels=boxes[inds],scores[inds],labels[inds]

            keep=batched_nms(boxes,scores,labels,self.nms_thresh)
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels









    def forward(self,
                features, #type:Dict[str,Tensor]
                proposals, #type:List[Tensor]
                image_shapes, #type:List[Tuple[int,int]]
                target, #type:Optional[List[Dict[str,Tensor]]]
                ):
        if self.training:
            proposals,labels,regression_targets=self.select_training_samples(proposals,target)
        else:
            labels=None
            regression_targets=None
        box_features=self.box_roi_pool(features,proposals,image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result=[]
        losses={}
        if self.training:
            loss_classifier,loss_box_reg=faster_cnn_loss(class_logits,box_regression,labels,regression_targets)
            losses={
                'loss_classifier':loss_classifier,
                'loss_box_reg':loss_box_reg
            }
        else:
            boxes,scores,labels=self.postprocess(class_logits,box_regression,proposals,image_shapes)


