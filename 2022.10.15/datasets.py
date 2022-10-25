from torch.utils.data import  Dataset
import random
import numpy as np
import os
from pathlib import Path
import tqdm
from PIL import Image

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break

def exif_size(img):
    s=img.size
    try:
        rotation=dict(img._getexif().items())[orientation]
        if rotation==6:
            s=(s[1],s[0])
        elif rotation==8:
            s=[s[1],s[0]]
    except:
        pass
    return s

class LoadImagesAndLabels(Dataset):
    def __init__(self,
                 path,   #data/my_train_data.txt or data/my_val_data.txt
                 img_size=416,   #
                 batch_size=16,
                 augument=False,
                 hyp=None,
                 rect=False,
                 rank=-1,
                 single_cls=False,
                 pad=0.0,
                 cached_images=False,
                 ):
        path=str(Path(path))
        if os.path.isfile(path):
            with open(path) as f:
                f=f.read().splitlines()
        else:
            raise Exception("%s does not exist"%path)
        img_formats=['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
        self.img_files=[x for x in f if os.path.splitext(x)[-1].lower() in img_formats]

        n=len(self.img_files)
        bi=np.floor(np.arange(n)/batch_size).astype(np.int)
        nb=bi[-1]+1

        self.n=n
        self.batch=bi
        self.img_size=img_size
        self.augment=augument
        self.hyp=hyp
        self.rect=rect
        self.mosaic=self.augment and not self.rect
        self.labels=[x.replace("images","labels").replace(os.path.splitext(x)[-1],".txt") for x in self.img_files]

        sp=path.replace(".txt",".shapes")
        try:
            with open(sp) as f:
                s=[x.split() for x in f.read().splitlines()]
        except Exception as e:
            if rank in [0,-1]:
                image_files=tqdm(self.img_files,desc="reading")
            else:
                image_files=self.img_files
            s=[exif_size(Image.open(f)) for f in image_files]
            np.savetxt(sp,s,fmt="%g")

        self.shapes=np.array(s)

        if self.rect:
            s=self.shapes
            ar=s[:,1]/s[:,0]
            irect=ar.argsort()
            self.img_files=self.img_files[irect]
            self.labels_files=self.labels_files[irect]
            self.shapes=s[irect]
            ar=ar[irect]

        shapes=[[1,1]]*nb
        for i in range(nb):
            ari=ar[i==bi]   #把排序后的第i个batch取出
            maxr,minr=max(ari),min(ari)

            if maxr<1:
                shapes[i]=[maxr,1]
            elif minr>1:
                shapes[i]=[1,1/minr]
        self.batch_shapes=np.ceil(np.array(shapes)*img_size/32).astype(np.int)*32

        if rank in [-1,0]:
            pbar=self.labels_files
        nm, nf, ne, nd = 0, 0, 0, 0
        for i,file in enumerate(pbar):
            try:
                with open(file) as f:
                    l=np.array([x.split() for x in f.read().splitlines()])
            except Exception as e:
                print("An error occurred while loading the file {}: {}".format(file, e))
                nm += 1  # file missing
                continue

            if l.shape[0]:
                assert  len(l.shape)==5

                self.labels[i]=l
                nf+=1
            else:
                ne+=1

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        hyp=self.hyp
        if self.mosaic:
            img,labels=load_mosaic(self,item)
            shapes=None
        else:
            img,(h0,w0),(h,w)=load_image(self,item)
            shape=self.batch_shapes[self.batch[item]] if self.rect else self.img_size
            img,ratio,pad=letterbox(img,shape,scale_up=self.augment)
            shapes=(h0,w0),((h/h0,w/w0).pad)

            lables=[]
            x=self.labels[item]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()  # label: class, x, y, w, h
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]






def load_mosaic(self,index):
    labels4=[]
    s=self.img_size
    xc,yc=[int(random.uniform(s*0.5,s*1.5)) for _ in range(2)]
    indices=[index]+[random.randint(0,len(self.labels)-1) for _ in range(3)]
    for i,index in enumerate(indices):
        img,_,(h,w)=laod_image(self,index)
        if i == 0:  # top left
            # 创建马赛克图像
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
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
    img4[y1a:y2a,x1a:x2a]=img[y1b:y2b,x1b:x2b]
    padw = x1a - x1b
    padh = y1a - y1b

    x = self.labels[index]
    labels = x.copy()  # 深拷贝，防止修改原数据
    if x.size > 0:  # Normalized xywh to pixel xyxy format
        # 计算标注数据在马赛克图像中的坐标(绝对坐标)
        #*w,*h是什么意思，这个地方感觉代码有问题
        labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw  # xmin
        labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh  # ymin
        labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw  # xmax
        labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh  # ymax
    labels4.append(labels)

    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # 设置上下限防止越界
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    return img4, labels4




