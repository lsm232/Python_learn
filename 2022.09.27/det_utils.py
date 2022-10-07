import torch
import math

class BoxCoder(object):

    def __init__(self,weights,bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode_single(self,rel_codes,boxes):
        widths=boxes[:,2]-boxes[:,0]
        heights=boxes[:,3]-boxes[:,1]
        ctr_x=boxes[:,0]+widths/2
        ctr_y=boxes[:,1]+heights/2

        wx,wy,ww,wh=self.weights
        dx=rel_codes[:,0::4]/wx
        dy=rel_codes[:,1::4]/wy
        dw=rel_codes[:,2::4]/ww
        dh=rel_codes[:,3::4]/wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x=dx*widths[:,None]+ctr_x[:,None]
        pred_ctr_y=dy*widths[:,None]+ctr_y[:,None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_xmin=pred_ctr_x-0.5*pred_w
        pred_ymin=pred_ctr_y-0.5*pred_h
        pred_xmax=pred_ctr_x+0.5*pred_w
        pred_ymax=pred_ctr_y+0.5*pred_h
        pred_boxes=torch.stack([pred_xmin,pred_ymin,pred_xmax,pred_ymax],dim=2).flatten(1)
        return pred_boxes






    def decode(self,pred_bbox_deltas,anchors_batch):
        anchors_per_image=[a.shape[0] for a in anchors_batch]
        concat_boxes=torch.cat(anchors_batch,dim=0)

        box_sum=concat_boxes.shape[0]
        pred_boxes=self.decode_single(pred_bbox_deltas,concat_boxes)

        if box_sum>0:
            pred_boxes=pred_boxes.reshape(box_sum,-1,4)
            return pred_boxes







class Mather(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    def __init__(self,high_threshhold,low_threshold,allow_low_quality_matches=False):
        self.BETWEEN_THRESHOLDS=-2
        self.BELOW_LOW_THRESHOLD=-1
        assert low_threshold<high_threshhold
        self.high_threshold=high_threshhold
        self.low_threshold=low_threshold
        self.allow_low_quality_matches=allow_low_quality_matches



class BalancedPositiveNegativeSampler(object):
    def __init__(self,batch_size_per_image,positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction