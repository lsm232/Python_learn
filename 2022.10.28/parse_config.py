import torch
import numpy as np
import os

def parse_data_cfg(data):
    path=data

    assert os.path.exists(path),f'{path} dose not exit'
    with open(path) as fid:
        lines=fid.readlines()

    options={}
    for line in lines:
        key=line.split("=")[0].strip()
        value=line.split("=")[1].strip()
        options[key]=value

    return options


def parse_model_cfg(cfg):
    assert os.path.exists(cfg),f"{cfg} dose not exit"

    with open(cfg,encoding='utf-8') as f:
        lines=[line.strip() for line in f.readlines()]
        lines=[line for line in lines if not line.startswith("#") and line]

        nets=[]
        for line in lines:
            if line.startswith("["):
                nets.append({})
                nets[-1]["type"]=line[1:-1]
                if nets[-1]["type"] == "convolutional":  #因为yolo predictor也有
                    nets[-1]["batch_normalize"] = 0

            else:
                key,value=line.split("=")
                key=key.strip()
                value=value.strip()
                if key=="anchors":
                    value=value.replace(" ","")
                    value=[float(x) for x in value.split(",")]
                    nets[-1][key]=np.array(value).reshape(-1,2)

                elif (key in ['from','layers','mask']) or (key=='size' and ',' in value):
                    nets[-1][key]=[int(x) for x in value.split(',')]

                else:
                    if value.isnumeric():
                        nets[-1][key]=int(value) if (int(value)-float(value))==0 else float(value)
                    else:
                        nets[-1][key]=value
    return nets





