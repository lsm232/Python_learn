import os
import torch
import torch.nn as nn

def check_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise ValueError(f"not find {path}")

def warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor):
    def f(x):
        if x>=warmup_iters:
            return 1
        else:
            alpha=float(x)/warmup_iters
            return warmup_factor*(1-alpha)+alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=f)

def wh_iou(wh1,wh2):
    wh1=wh1[:,None]
    wh2=wh2[None,:]
    inter=torch.min(wh1,wh2).prod(2)
    iou=inter/(wh1.prod(2)+wh2.prod(2)-inter)
    return iou


def build_targets(p,targets,model):
    nt=targets.shape[0]  #一个batch有多少个gt box
    tcls,tbox,indices,anch=[],[],[],[]
    gain=torch.ones(6)

    multi_gpu=type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i,j in enumerate(model.yolo_layers):
        anchors=model.mudule.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:]=torch.tensor(p[i].shape)[[3,2,3,2]]
        na=anchors.shape[0]
        at=torch.arange(na).view(na,1).repeat(1,nt)

        a,t,offsets=[],targets*gain,0  #targets是相对坐标，要把它转换为特征层上的绝对坐标

        if nt:
            j=wh_iou(anchors,t[:,4:6])>model.hyp['iou_t']
            a,t=at[j],t.repeat(na,1,1)[j]

        b,c=t[:,:2].long().T
        gxy=t[:,2:4]
        gwh=t[:,4:6]
        gij = (gxy - offsets).long()
        gi,gj=gij.t()

        indices.append((b,a,gj.clamp_(0,gain[3]-1),gi.clamp_(0,gain[2]-1)))  #这里为什么要写成先y再x
        tbox.append(torch.cat((gxy-gij,gwh),1))
        anch.append(anchors[a])
        tcls.append(c)

    return tcls, tbox, indices, anch


class FocalLoss(nn.Module):
    def __init__(self,loss_fn,gamma=1.5,alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fn=loss_fn
        self.gamma=gamma
        self.alpha=alpha
        self.reduction=loss_fn.reduction
        self.loss_fcn.reduction = 'none'
    def forward(self,pred,true):
        loss=self.loss_fn(pred,true)
        pred_prob=torch.sigmoid(pred)
        p_t=true*pred_prob+(1-true)(1-pred_prob)
        alpha_factor=true*self.alpha+(1-true)*(1-self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



def compute_loss(p,targets,model):
    device=p[0].device
    lcls=torch.zeros(1)
    lbox=torch.zeros(1)
    lobj=torch.zeros(1)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)

    h=model.hyp
    red="mean"
    BCEcls=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']],device=device),reduction=red)
    BCEobj=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["pos_pw"]],device=device),reduction=red)

    g=h['fl_gamma']
    if g>0:
        BCEcls,BCEobj=FocalLoss(BCEcls,g),FocalLoss(BCEobj,g)

    for i,pi in enumerate(p):
        b,a,gj,gi=indices[i]
        tobj=torch.zeros_like(pi[...,0],device=device)

        nb=b.shape[0]
        if nb:
            ps=pi[b,a,gj,gi]
            pxy = ps[:, :2].sigmoid()
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Obj
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

        # 乘上每种损失的对应权重
        lbox *= h['giou']
        lobj *= h['obj']
        lcls *= h['cls']

        # loss = lbox + lobj + lcls
        return {"box_loss": lbox,
                "obj_loss": lobj,
                "class_loss": lcls}



