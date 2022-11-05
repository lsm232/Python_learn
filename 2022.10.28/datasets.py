from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os
import random
import cv2


def load_image(self,index):
    img_path=self.img_files[index]
    img=cv2.imread(img_path)
    h0,w0=img.shape[:2]
    r=self.img_size/max(h0,w0)
    if r!=1:
        interp=cv2.INTER_AREA if r<1 and not self.augument else cv2.INTER_LINEAR
        img=cv2.resize(img,(int(h0*r),int(w0*r)),interpolation=interp)
    return img,(h0,w0),img.shape[:2]




def load_mosaic(self,index):
    labels4=[]
    s=self.img_size

    #获取生成一个顶点
    xc,yc=[int(random.uniform(0.5*s,1.5*s)) for _ in range(2)]
    #再获取三张图像用于拼接
    indices=[index]+[random.randint(0,len(self.label_files)-1) for _ in range(3)]
    #遍历四张图像并拼接
    for i,item in enumerate(indices):
        img,_,(h,w)=load_image(self,item)

        #如果是第一张图象，先创建一个masaic图像
        if i==1:
            imgs4=np.full(shape=(s,s,img.shape[2]),fill_value=114,dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        imgs4[y1a:y2a,x1a:x2a]=img[y1b:y2b, x1b:x2b]

        padw=x1a-x1b
        padh=y1a-y1b

        x=self.labels[item]
        labels=x.copy()

        if x.size>0:
            labels[:,1]=w*(labels[:,1]-labels[:,3]/2)+padw
            labels[:,2]=h*(labels[:,2]-labels[:,4]/2)*padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw  # xmax
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh  # ymax
        labels4.append(labels)

    if len(labels4):
        labels4=np.concatenate(labels4,0)
        np.clip(labels4[:,1:],0,2*s,out=labels4[:,1:])  #0为label

    return imgs4, labels4





class LoadImageAndLabels(Dataset):
    def __init__(self,
                 path, #data/my_train_data.txt  or data/my_val_data.txt
                 img_size=416,
                 batch_size=2,
                 augment=False,
                 hyp=None,
                 rect=False,
                 single_cls=False,



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

                    if l.shape[0]:
                        if np.unique(l,axis=0).shape[0]<l.shape[0]:
                            nd+=1
                        if single_cls:
                            l[:,0]=1

                        self.labels[i]=l
                        nf+=1

                    else:
                        ne+=1
    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        hyp=self.hyp

        if self.mosaic:
            img,labels=load_mosaic(self,index)
            shapes=None
        else:
            img,(h0,w0),(h,w)=load_image(self,index)
            shape=self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augument,scale_fill=False)

            shapes = (h0, w0), ((h / h0, w / w0), pad)

            




def letterbox(img,new_shape,scale_up,auto,scale_fill):
    shape=img.shape[:2]
    h,w=img.shape[:2]
    r=min(h/new_shape[0],w/new_shape[1])
    if not scale_up:
        r=min(1,r)
    ratio=r,r  #(h,w)
    new_unpad=int(ratio[0]*h),int(ratio[1]*w)
    dw,dh=new_shape[0]-new_unpad[0],new_shape[1]-new_unpad[1]

    if auto:
        dw,dh=np.mod(dw,32),np.mod(dh,32)
    elif scale_fill:
        dw,dh=0,0
        new_unpad=new_shape
        ratio=new_shape[0]/h,new_shape[1]/w

    dw/=2
    dh/=2

    if shape[::-1]!=new_unpad:
        img=cv2.resize(img,new_unpad)

    top,bottom=int(round(dh-0.1)),int(round(dh+0.1))
    left,right=int(round(dw-0.1)),int(round(dh+0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value='red')  # add border
    return img, ratio, (dw, dh)






















