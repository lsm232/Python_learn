import torch
import torchvision.transforms as transforms

import numpy as np
from torch.utils.data import DataLoader,Dataset



def load_dataset(root_dir,redux,params,shuffled=False,single=False):
    noise=(params.noise_type,params.noise_params)
    dataset=NoisyDataset(root_dir,redux,params.crop_size,clean_targets=params.clean_targets,noise_dist=noise,seed=params.seed)
    if single:
        return DataLoader(dataset=dataset,batch_size=1,shuffle=shuffled)
    else:
        return DataLoader(dataset=dataset,batch_size=params.batch_size,shuffle=shuffled)

 class AbstractDataset(Dataset):
    def __init__(self,root_dir,redux=0,crop_size=128,clean_targets=False):
        super(AbstractDataset, self).__init__()
        self.imgs=[]
        self.root_dir=root_dir
        self.redux=redux
        self.crop_size=crop_size
        self.clean_targets=clean_targets
    def _random_crop(self,img_list):
        #对所有的图像进行相同的crop
        w,h=img_list[0].size
        assert w>=self.crop_size and h>=self.crop_size,f'error :crop_size:{self.crop_size} and img_size:{w,h}'
        cropped_imgs=[]
        i=np.random.randint(0,h-self.crop_size+1)
        j=np.random.randint(0,w-self.crop_size+1)
        for img in img_list:
            if min(w,h)<self.crop_size:
                



