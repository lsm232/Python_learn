from torch.utils.data import Dataset
import torch
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
            labels=[]
            x=self.labels[index]
            if x.size>0:
                labels=x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]


        if self.augument:
            if not self.mosaic:
                img,labels=random_affine(img, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment colorspace
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

            nL=len(labels)
            if nL:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0-1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            if self.augment:
                # random left-right flip
                lr_flip = True  # 随机水平翻转
                if lr_flip and random.random() < 0.5:
                    img = np.fliplr(img)
                    if nL:
                        labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

                # random up-down flip
                ud_flip = False
                if ud_flip and random.random() < 0.5:
                    img = np.flipud(img)
                    if nL:
                        labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center

            labels_out = torch.zeros((nL, 6))  # nL: number of labels
            if nL:
                labels_out[:, 1:] = torch.from_numpy(labels)

            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self,index):
        o_shapes=self.shapes[index][::-1]
        x=self.labels[index]
        labels=x.copy()
        return torch.from_numpy(labels),o_shapes

    @staticmethod
    def collate_fn(batch):
        img,label,path,shapes,index=zip(*batch)
        for i,j in enumerate(label):
            j[:,0]=i
        return torch.stack(img,0),torch.cat(label,0),path,shapes,index




def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """随机旋转，缩放，平移以及错切"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # 这里可以参考我写的博文: https://blog.csdn.net/qq_37541097/article/details/119420860
    # targets = [cls, xyxy]

    # 最终输出的图像尺寸，等于img4.shape / 2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 进行仿射变化
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    # 这里可以参考我写的博文:https://blog.csdn.net/qq_37541097/article/details/119478023
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed




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
        ratio=new_shape[1]/w,new_shape[0]/h

    dw/=2
    dh/=2

    if shape[::-1]!=new_unpad:
        img=cv2.resize(img,new_unpad)

    top,bottom=int(round(dh-0.1)),int(round(dh+0.1))
    left,right=int(round(dw-0.1)),int(round(dh+0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value='red')  # add border
    return img, ratio, (dw, dh)






















