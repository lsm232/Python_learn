import torch
import torch.nn as nn
import math
from distributed_utils import *


def wh_iou(wh1,wh2):
    wh1=wh1[:,None]
    wh2=wh2[None]
    inter=torch.min(wh1,wh2).prod(2)
    return inter/(wh1.prod(2)+wh2.prod(2)-inter)

def build_targets(p,targets,model):
    nt=targets.shape[0]  #有多少个物体
    tcls,tbox,indices,anch=[],[],[],[]
    gain=torch.ones(6)

    multi_gpu=type(model) in (nn.parallel.DataParallel,nn.parallel.DistributedDataParallel)
    for i,j in enumerate(model.yolo_layers):
        anchors=model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:]=torch.tensor(p[i].shape)[[3,2,3,2]]

        na=anchors.shape[0]
        at=torch.arange(0,na).view(na,1).repeat(1,nt)

        a,t,offsets=[],targets*gain,0
        if nt:
            j=wh_iou(anchors,t[:,4:6])> model.hyp['iou_t']
            a,t=at[j],t.repeat(na,1,1)[j]

        b,c=t[:,:2].long().T
        gxy=t[:,2:4]
        gwh=t[:,4:6]
        gij=(gxy-offsets).long()
        gi,gj=gij.T

        indices.append((b,a,gj.clamp_(0,gain[3]-1),gi.clamp_(0,gain[2]-1)))
        tbox.append(torch.cat((gxy-gij,gwh),1))
        anch.append(anchors[a])
        tcls.append(c)

    return tcls,tbox,indices,anch




def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def bbox_iou(box1,box2,x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    box2=box2.t()
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1,b1_x2=box1[0]-box1[2]/2,box1[0]+box1[2]/2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter=(torch.min(b1_x2,b2_x2)-torch.max(b1_x1,b1_x2)).clamp(0)* (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou=inter/union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou




def compute_loss(p,targets,model):  #这里的p对应三层特征图的预测结果，每个特征图上每个像素点有25个预测值
    device=p[0].device
    lcls=torch.zeros(1,device=device)
    lbox=torch.zeros(1,device=device)
    lobj=torch.zeros(1,device=device)
    tcls,tbox,indices,anchors=build_targets(p, targets, model)
    h=model.hyp
    red='mean'

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction=red)

    cp, cn = smooth_BCE(eps=0.0)
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    for i,pi in enumerate(p):
        b,a,gj,gi=indices[i]
        tobj=torch.zeros_like(pi[...,0],device=device)

        nb=b.shape[0]
        if nb:
            ps=pi[b,a,gj,gi]
            pxy=ps[:,:2].sigmoid()
            pwh=ps[:,2:4].exp()*anchors[i]
            pbox=torch.cat([pxy,pwh],1)
            giou=bbox_iou(pbox.t(),tbox[i],x1y1x2y2=False, GIoU=True)
            lbox += (1.0 - giou).mean()

            # Obj
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            if model.nc>1:
                t=torch.full_like(ps[:,5:],cn,device=device)
                t[range(nb),tcls[i]]=cp
                lcls+=BCEcls(ps[:,5:],t)

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        lbox *= h['giou']
        lobj *= h['obj']
        lcls *= h['cls']

        # loss = lbox + lobj + lcls
        return {"box_loss": lbox,
                "obj_loss": lobj,
                "class_loss": lcls}








