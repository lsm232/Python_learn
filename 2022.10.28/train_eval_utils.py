import torch
import torch.nn.functional as F
import math
import random
from torch.cuda import amp

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq,accumulate,img_size,grid_min,grid_max,gs,muti_scale,warmup,scaler):
    model.train()
    lr_scheduler=None

    if epoch==0 and warmup==True:
        warmup_factor=1/1000
        warmup_iters=min(1000,len(data_loader)-1)

        lr_scheduler=warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor)
        accumulate=1

    mloss=torch.zeros(4).to(device)
    now_lr=0.
    nb=len(data_loader)

    for i,(imgs,targets,paths,_,_) in enumerate(data_loader):
        ni=i+nb*epoch
        imgs=imgs.to(device)/255.
        targets=targets.to(device)

        if muti_scale:
            if ni%accumulate==0:
                img_size=random.randrange(grid_min,grid_max+1)*gs
            sf=img_size/max(imgs.shape[2:])

            if sf!=1:
                ns=[math.ceil(x*sf/gs)*gs  for x in imgs.shape[2:]]
                imgs=F.interpolate(imgs,size=ns)

        with amp.autocast(enabled=scaler is not None):
            pred=model(imgs)
            loss_dict=compute_loss(pred,targets,model)


