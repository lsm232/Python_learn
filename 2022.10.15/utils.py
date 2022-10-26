import os
import torch

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

