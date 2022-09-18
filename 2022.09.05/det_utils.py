import torch
import math

class BoxCoder(object):

    def __init__(self,weights):
        self.weights=weights


    def decode_single(self,rel_codes,boxes):
        widths=boxes[:,-2]-boxes[:,1]
        heights=boxes[:,-1]-boxes[:,0]
        c_x=boxes[:,0]+0.5*widths
        c_y=boxes[:,1]+0.5*heights

        wx, wy, ww, wh = self.weights
        dx=rel_codes[:,0::4]/wx
        dy=rel_codes[:,1::4]/wy
        dw=rel_codes[:,2::4]/ww
        dh=rel_codes[:,3::4]/wh

        self.bbox_xform_clip=math.log(1000. / 16)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_c_x=dx*widths[:,None]+c_x[:,None]
        pred_c_y=dy*heights[:,None]+c_y[:,None]
        pred_w=torch.exp(dw)*widths[:,None]
        pred_h=torch.exp(dh)*heights[:,None]

        pred_boxes_xmin=pred_c_x-0.5*pred_w
        pred_boxes_ymin=pred_c_y-0.5*pred_h
        pred_boxes_xmax=pred_c_x+0.5*pred_w
        pred_boxes_ymax=pred_c_y+0.5*pred_h
        pred_boxes=torch.stack([pred_boxes_xmin,pred_boxes_ymin,pred_boxes_xmax,pred_boxes_ymax],dim=2).flatten(1)
        return pred_boxes







    def decode(self,rel_codes,boxes):
        boxes_per_image=[b.shape[0] for b in boxes]
        concat_boxes=torch.cat(boxes,dim=0)
        box_sum=concat_boxes.shape[0]

        pred_boxes=self.decode_single(rel_codes,boxes)