import torch
import torch.nn as nn

def build_targets(p,targets,model):
    nt=targets.shape[0]  #有多少个物体
    tcls,tbox,indices,anch=[],[],[],[]
    gain=torch.ones(6)

    multi_gpu=type(model) in (nn.parallel.DataParallel,nn.parallel.DistributedDataParallel)




def compute_loss(p,targets,model):  #这里的p对应三层特征图的预测结果，每个特征图上每个像素点有25个预测值
    device=p[0].device
    lcls=torch.zeros(1,device=device)
    lbox=torch.zeros(1,device=device)
    lobj=torch.zeros(1,device=device)
    tcls,tbox,indices,anchors=build_targets(p, targets, model)

