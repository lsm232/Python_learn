from .unet import *
import datetime
import torch
import torch.nn as nn

class noise2noise(object):
    def __init__(self,params,trainable):
        self.p=params
        self.trainable=trainable
        self._compile()

    def _compile(self):
        self.model=UNet(1,1)
        if self.trainable:
            self.optim=torch.optim.Adam(
                self.model.parameters(),
                lr=0.01,
                betas=self.p.adam[:2],
                eps=self.p.adam[2]),
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim,patience=4,factor=0.5)
            self.loss=nn.L1Loss()
        self.use_cuda =torch.cuda.is_available()
        if self.use_cuda:
            self.model=self.model.cuda()
            if self.trainable:
                self.loss=self.loss.cuda()

    def _print_params(self):
        print('training parameters')
        self.p.cuda=self.use_cuda
        param_dict=vars(self.p)
        pretty=lambda x:x.replace('_',' ').captilaize()
        print('\n'.join(' {}={}'.format(pretty(k),str(v)) for k,v in param_dict.items()))
        print()

    def save_model(self,epoch,stats,first=False):
        if True:
            if self.p.clean_targets:
                ckpt_dir_name=f'{datetime.now():{self.p.noise_type}-clean-%m%d%H}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-noise-%m%d%H}'

            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name=f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name=self.p.noise_type

            if self.p.ckpt_overwrite:







