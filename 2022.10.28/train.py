import argparse
from mode
import yaml
import torch
import datetime
import math
from .parse_config import *

def train(hyp):
    device=torch.device("gpu" if torch.cuda.is_available() else "cpu")
    print(f"using {device.type} training......")

    results_file="results_file_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    gs=32
    assert math.fmod(imgsz_test,gs)==0,"in test,img_size must be %g mutiple"%(gs)
    grid_min,grid_max=imgsz_test//gs,imgsz_test//gs

    if multi_scale:
        imgsz_max=imgsz_train//0.667
        imgsz_min=imgsz_train//1.5
        grid_max=imgsz_max//gs
        grid_min=imgsz_min//gs
        imgsz_max,imgsz_min=int(grid_max*gs),int(grid_min*gs)
        imgsz_train=imgsz_max
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    data_dict=parse_data_cfg(data)
    train_path=data_dict['train']
    test_path=data_dict['test']
    nc=1 if opt.single_cls else int(data_dict['classes'])

    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    model=Darknet(cfg).to(device)












if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cfg', type=str, default='cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')  #命令行设置--rect，则启用，否则默认关闭
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str,
                        default='G:\leran to play\my_yolo_dataset\yolo_pretrain_weights/yolov3-spp-ultralytics-512.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    opt=parser.parse_args()


    #读取超参文件
    with open(opt.hyp) as fid:
        hyp=yaml.load(fid,Loader=yaml.FullLoader)    #以字典的形式读入，类似json.load

    train(hyp)



