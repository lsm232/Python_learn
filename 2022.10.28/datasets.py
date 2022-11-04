from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os

class LoadImageAndLabels(Dataset):
    def __init__(self,
                 path, #data/my_train_data.txt  or data/my_val_data.txt
                 img_size=416,
                 batch_size=2,
                 augment=False,
                 hyp=None,
                 rect=False,



                 ):
        try:
            path=str(Path(path))
            if os.path.isfile(path):
                with open(path,'r') as fid:
                    lines=fid.readlines()
            else:
                raise Exception("{} dose not exit".format(path))
            img_formats=['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            self.img_files=[x.strip() for x in lines if os.path.splitext(x.strip())[-1].lower() in img_formats]
            self.img_files.sort()

        except:
            raise Exception("{} dose not exit".format(path))

        n=len(self.img_files)
        assert n>0,"not found any image"

        bi=np.floor(np.arange(n)/batch_size).astype(np.int)
        nb=bi[-1]+1

        self.n=n
        self.batch=bi
        self.img_size=img_size
        self.augument=augment
        self.hyp=hyp
        self.rect=rect
        self.mosaic=self.augument and not self.rect

        self.label_files=[x.replace("images","labels").replace(os.path.splitext(x)[-1],'.txt') for x in self.img_files]

        sp=path.replace(".txt",".shapes")

        assert os.path.exists(sp),print("{} does not exit".format(sp))

        with open(sp) as fid:
            s=[x.split() for x in fid.read().splitlines()]
            assert len(s)==n

        self.shapes=np.array(s,dtype=np.float32)

        if self.rect:
            s=self.shapes
            ar=s[:,1]/s[:,0]
            irect=ar.argsort()  #返回的是index，从小到大

            self.img_files=[self.img_files[i] for i in irect]
            self.label_files=[self.label_files[i] for i in irect]
            self.shapes=s[rect]
            ar=ar[rect]

            shapes=[[1,1]]*nb
            for i in range(nb):
                ari=ar[bi==i]
                mini,maxi=ari.min(),ari.max()

                if maxi<1:
                    shapes[i]=[maxi,1]
                elif mini>1:
                    shapes[i]=[1,1/mini]
                self.batch_shapes=np.ceil(np.asarray(shapes)*img_size/32).astype(np.int)*32

            self.imgs=[None]*n
            self.labels=[np.zeros((0,5),dtype=np.float32)]*n
            extract_bounding_boxes, labels_loaded = False, False
            nm,nf,ne,nd=0,0,0,0

            if rect is True:
                np_labels_path=str(Path(self.label_files[0]).parent)+'.rect.npy'
            else:
                np_labels_path=str(Path(self.label_files[0]).parent)+'.norect.npy'

            if os.path.isfile(np_labels_path):
                x=np.load(np_labels_path)
                if len(x)==n:
                    self.labels=n
                    labels_loaded=True

            pbar=self.label_files

            for i,file in enumerate(pbar):
                if labels_loaded is True:
                    l=self.labels[i]
                else:
                    try:
                        with open(file,"r") as f:
                            l=np.array([x.split() for x in f.read().splitlines()],dtype=np.float32)
                    except Exception as e:
                        print("An error occurred while loading the file {}: {}".format(file, e))
                        nm += 1  # file missing
                        continue

                














