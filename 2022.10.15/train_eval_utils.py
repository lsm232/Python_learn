import torch
import math
from .utils import *
import random
import torch.nn.functional as F
from torch.cuda import amp


def train_one_epoch(model, optimizer, data_loader, img_size, epoch, warmup,device,accumulate,multi_scale,grid_min,grid_max,gs,scaler):
    model.train()

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(len(data_loader), 1000)

        lr_scheduler =warmup_lr_scheduler(optimizer,warmup_factor,warmup_iters)
        accumulate=1

    mloss=torch.zeros(4).to(device)
    now_lr=0.
    nb=len(data_loader)

    for i,(imgs,targets,paths,_,_) in enumerate(data_loader):
        ni=len(data_loader)*epoch+i
        imgs=imgs.to(device).float()/255.
        targets=targets.to(device)

        if multi_scale:
            if ni% accumulate ==0:
                img_size=random.randrange(grid_min,grid_max+1)*gs
            sf=img_size/max(imgs)


        if sf!=1:
            ns=[math.ceil(x*sf/gs)*gs for x in imgs.shapes[2:]]
            imgs=F.interpolate(imgs,size=ns)


        with amp.autocast(enabled=scaler is not None):
            pred=model(imgs)
            loss_dict = compute_loss(pred, targets, model)
            losses = sum(loss for loss in loss_dict.values())












