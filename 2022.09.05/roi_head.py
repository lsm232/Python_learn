import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional,List,Dict,Tuple,Optional
from .boxes import *
from .det_utils import *


class ROIHeads(nn.Module):
    def __init__(self,batch_size_per_image,positive_fraction):
        self.proposal_matcher=Mather
        self.fg_bg_sampler=BalancedPositiveNegativeSampler(batch_size_per_image,positive_fraction)


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
