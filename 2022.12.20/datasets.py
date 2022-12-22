import torch
from string import ascii_letters
import random
from PIL import Image,ImageFont,ImageDraw
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf

import numpy as np
from torch.utils.data import DataLoader,Dataset
from sys import platform



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
        self.clean_targets=clean_targets
        self.crop_size=crop_size
    def _random_crop(self,img_list):
        #对所有的图像进行相同的crop
        w,h=img_list[0].size
        assert w>=self.crop_size and h>=self.crop_size,f'error :crop_size:{self.crop_size} and img_size:{w,h}'
        cropped_imgs=[]
        i=np.random.randint(0,h-self.crop_size+1)
        j=np.random.randint(0,w-self.crop_size+1)
        for img in img_list:
            if min(w,h)<self.crop_size:
                img=transforms.Resize(img,(self.crop_size,self.crop_size))
            cropped_imgs.append(ttf.crop(img,i,j,self.crop_size,self.crop_size))
        return cropped_imgs

    def __getitem__(self, item):
        raise NotImplementedError('Abstract method not implemented!')

    def __len__(self):
        return len(self.imgs)

class NoisyDataset(AbstractDataset):
    def __init__(self,root_dir,redux,crop_size,clean_targets=False,noise_dist=('gaussian',50.),seed=None):
        super(NoisyDataset, self).__init__()
        self.imgs=os.listdir(root_dir)
        if redux:
            self.imgs=self.imgs[:redux]

        self.noise_type=noise_dist[0]
        self.noise_param=noise_dist[1]
        self.seed=seed
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self,img):
        w,h=img.size
        c=len(img.getbands())

        if self.noise_type=='poisson':
            noise=np.random.poisson(img)
            noise_img=img+noise
            noise_img=255*(noise_img/np.amax(noise_img))
        else:
            if self.seed:
                std=self.noise_param
            else:
                std=np.random.uniform(0,self.noise_param)
            noise=np.random.normal(0,std,(h,w,c))
            noise_img=np.expand_dims(np.array(img),axis=2)+noise

        noise_img=np.clip(noise_img,-1,1).astype(np.float32)
        return Image.fromarray(noise_img[...,0])

    def _add_text_overlay(self,img):
        assert self.noise_param<1
        w,h=img.size
        c=len(img.getbands())

        if platform=='linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        text_img=img.copy()
        text_draw=ImageDraw.Draw(text_img)

        w,h=img.size
        mask_img=Image.new('1',(w,h))
        mask_draw=ImageDraw.Draw(mask_img)

        if self.seed:
            random.seed(self.seed)
            max_occupancy=self.noise_param
        else:
            max_occupancy=np.random.uniform(0,self.noise_param)

        def get_occupancy(x):
            y=np.array(x,dtype=np.uint8)
            return np.sum(y)/y.size

        while 1:
            font=ImageFont.truetype(serif,np.random.randint(16,21))
            length=np.random.randint(10,25)
            chars=''.join(random.choice(ascii_letters) for i in range(length))
            color=np.random.randint(0,255,c)
            pos=(np.random.randint(0,w),np.random.randint(0,h))
            text_draw.text(pos,chars,color,font=font)

            mask_draw.text(pos,chars,1,font=font)
            if get_occupancy(mask_img)>max_occupancy:
                break
        return text_img

    def _corrupt(self,img):
        if self.noise_type in ['gaussian','poisson']:
            return self._add_noise(img)
        elif self.noise_type=='text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))

    def __getitem__(self, item):
        img_path=os.path.join(self.root_dir,self.imgs[item])
        img=Image.open(img_path).convert("F")
        if self.crop_size!=0:
            img=self._random_crop(img)
        source=ttf.to_tensor(self._corrupt(img))

        if self.clean_targets:
            target=ttf.to_tensor(img)
        else:
            target=ttf.to_tensor(self._corrupt(img))

        return source, target





        



