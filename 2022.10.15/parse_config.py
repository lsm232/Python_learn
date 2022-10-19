import os
import numpy as np

def parse_data_cfg(path):
    assert os.path.exists(path),f"{path} not found"
    with open(path) as f:
        lines=f.readlines()
    options={}
    for line in lines:
        line=line.strip()
        if line=='' or line.startswith("#"):
            continue
        key,value=line.split("=")
        options[key]=value
    return options

def parse_model_cfg(path):
    assert os.path.exists(path),f"{path} not found"
    with open(path) as f:
        lines=f.read().split("\n")
    lines=[x for x in lines if x and not x.startswith("#")]
    lines=[line.strip() for line in lines]

    mdefs=[]
    for line in lines:
        if line.startswith("["):
            mdefs.append({})
            mdefs[-1]["type"]=line[1:-1]
            if mdefs[-1]["type"]=="convolutional":
                mdefs[-1]["batch_normalize"]=0
        else:
            key,value=line.split("=")
            key=key.strip()
            value=value.strip()
            if "anchors"==key:
                value=value.replace(" ","")
                mdefs[-1][key]=np.asarray([float(x) for x in value.split(",")]).reshape(-1,2)
            elif (key in ["from","layers","mask"]) or (key=="size" and "," in value):
                mdefs[-1][key]=[int(x) for x in value.split(",")]
            else:
                if value.isnumeric():
                    mdefs[-1][key]=int(value) if (int(value)-float(value)==0) else float(value)
                else:
                    mdefs[-1][key]=value

    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    for x in mdefs[1:]:
        for k in x:
            if k not in supported:
                raise ValueError("不支持的{}".format(k))
    return mdefs






    c=1



# parse_data_cfg('./data/my_data.data')

parse_model_cfg("./cfg/my_yolov3.cfg")