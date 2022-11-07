import torch
import torch.nn as nn


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

    

