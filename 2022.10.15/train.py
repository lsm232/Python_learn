import torch
import glob
import torch.nn as nn
import argparse
import os
from .utils import *
import yaml
from torch.utils.tensorboard import SummaryWriter
import datetime
from .parse_config import *
from models import *


def train(hyp):
    device=torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"use device: {device}")

    wdir="weights"+os.sep
    best=wdir+"best.pt"
    results_file="results{}.txt".format(datetime.datetime.now().strftime("%Y%M%D-%H%M%S"))

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)  当显存不够时，又想使用大的batch-size，则将损失函数累积到的想要的batch_size再更新

    #图像尺寸应该为32的整数倍
    gs=32
    assert imgsz_test%gs==0,"--img_size {} must be a {}-multiple".format(imgsz_test,gs)
    grid_min,grid_max=imgsz_test//gs,imgsz_train//gs
    if multi_scale:
        imgsz_min=opt.img_size//1.5
        imgsz_max=opt.img_size//0.667

        grid_min,grid_max=imgsz_min//gs,imgsz_max//gs
        imgsz_min,imgsz_max=int(grid_min*gs),int(grid_max*gs)
        imgsz_train=imgsz_max
        print("use muti-scale training,range [{} , {}]".format(imgsz_min,imgsz_max))

    data_dict=parse_data_cfg(data)
    train_path=data_dict["train"]
    test_path=data_dict["valid"]
    nc=1 if opt.sinle_cls else int(data_dict["classes"])
    #迷惑，不知道两个超参用来干嘛
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320


    #为什么要删除之前的结果了
    for f in glob.glob(results_file):
        os.remove(f)

    model=Darknet(cfg).to(device)









if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--cfg',type=str,default='./cfg/my_yolov3.cfg',help='.cfg path')
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')  #如果训练的时候设置--rect 则执行，不设置就不执行
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='yolo_ex', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    opt=parser.parse_args()

    check_file(opt.cfg)
    check_file(opt.data)
    check_file(opt.hyp)

    print(opt)

    with open(opt.hyp) as file:
        hyp=yaml.load(file,Loader=yaml.FullLoader)
    tb_writer=SummaryWriter(comment=opt.name)
    train(hyp)


