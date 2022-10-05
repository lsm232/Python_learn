import torch

def train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=50,warm_up=True,scaler=None):
    model.train()
    lr_scheduler=None

    if epoch==0 and warm_up is True:
        warm_up_factor=1.0/1000
        warmup_iters=min(10000,len(data_loader)-1)

        lr_scheduler=distributed_utils.warmup_lr_scheduler(optimizer,warmup_iters,warm_up_factor)
        for i,(images,targets) in enumerate(data_loader):
            images=list(image.to(device) for image in images)
            targets=[{k:v.to(device) for k,v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict=model(images,targets)