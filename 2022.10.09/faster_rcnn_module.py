import torch
import copy
import os
from torchvision.transforms import functional as F
import random
from torch.utils.data import Dataset
from lxml import etree
import json
from PIL import Image
import bisect


device=torch.device("gpu" if torch.cuda.is_available() else "cpu")

#
class Compose(object):
    def __init__(self,funs):
        self.funs=funs
    def __call__(self,images,targets):
        for fun in self.funs:
            images,targets=fun(images,targets)
        return images,targets
class ToTensor(object):
    """将pil image转换为tensor"""
    def __call__(self, images,targets):
        #return torch.as_tensor(images),torch.as_tensor(targets)
        return F.to_tensor(images),targets
class RandomHorizontalFlip(object):
    def __init__(self,prob=0.5):
        self.prob=prob
    def __call__(self,images,targets):
        if random.random()<self.prob:
            height,width=images.shape[-2:]
            images.flip(-1)
            bbox=targets["boxes"]
            bbox[:,[0,2]]=width-bbox[:,[2,0]]
            targets["boxes"]=bbox
        return images,targets
#
data_transform={
    "train":Compose([ToTensor(),RandomHorizontalFlip(0.5)]),
    "val":Compose([ToTensor()]),
}
#
class vocDataset(Dataset):
    def __init__(self, root=r'G:\leran to play\VOCdevkit\VOC2012', transforms=data_transform['train'], txt_name="train.txt"):
        self.root = root
        self.images_path = os.path.join(self.root, "JPEGImages")
        self.annotations_path = os.path.join(self.root, "Annotations")
        self.txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)

        assert os.path.exists(self.images_path), "not found {} file".format(self.images_path)
        assert os.path.exists(self.annotations_path), "not found {} file".format(self.annotations_path)
        assert os.path.exists(self.txt_path), "not found {} file".format(self.txt_path)

        xml_list = []
        self.xml_list=[]
        with open(self.txt_path) as fid:
            xml_list.extend(os.path.join(self.annotations_path, line.strip()+".xml") for line in fid.readlines())
            #xml_list=[os.path.join(self.annotations_path, line.strip()+".xml") for line in fid.readlines()]
        for xml_path in xml_list:
            with open(xml_path) as fid:
                xml_str=fid.read()
            xml=etree.fromstring(xml_str)
            xml_dict=self.parse_xml_to_dict(xml)
            data=xml_dict["annotation"]
            if "object" not in data:
                print(f"no object in {xml_path} ,skip")
                continue
            self.xml_list.append(xml_path)

        json_file=r'./pascal_voc_classes.json'
        with open(json_file) as fid:
            self.class_dict=json.load(fid)

        self.transforms=transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        xml_file=self.xml_list[item]
        with open(xml_file) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)['annotation']

        img_path=os.path.join(self.images_path,data["filename"])
        img=Image.open(img_path)

        targets={}
        targets["labels"]=[]
        targets["is_crowd"]=[]
        targets["boxes"]=[]
        for ob in data['object']:
            targets["labels"].append((self.class_dict[ob["name"]]))
            targets["is_crowd"].append(float(ob["difficult"]))
            targets["boxes"].append([float(ob["bndbox"]["xmin"]),float(ob["bndbox"]["ymin"]),float(ob["bndbox"]["xmax"]),float(ob["bndbox"]["ymax"])])

        targets['boxes']=torch.as_tensor(targets['boxes'])
        targets['is_crowd']=torch.as_tensor(targets['is_crowd'])
        targets['labels']=torch.as_tensor(targets['labels'])


        if self.transforms is not None:
            img,targets=self.transforms(img,targets)

        return img,targets

    def parse_xml_to_dict(self,xml):
        if len(xml)==0:
            return {xml.tag:xml.text}
        result={}
        for xml_child in xml:
            child_result=self.parse_xml_to_dict(xml_child)
            if (xml_child.tag!="object"):
                result[xml_child.tag]=child_result[xml_child.tag]
            else:
                if "object" not in result:
                    result["object"]=[]
                result["object"].append(child_result["object"])
        return {xml.tag:result}

    def get_height_and_width(self,idx):
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)
        height=int(data['annotation']['size']["height"])
        width=int(data['annotation']['size']["width"])
        return height,width

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
#

train_dataset=vocDataset()
train_sampler=None

#
def _compute_aspect_ratios_custom_dataset(dataset):
    file_nums=len(dataset)
    aspect_ratios=[]
    for i in range(file_nums):
        height,width=dataset.get_height_and_width(1)
        aspect_ratios.append(height/width)
    return aspect_ratios

def _quantize(x,bins):
    bins=copy.deepcopy(bins)
    bins=sorted(bins)
    quantized=list(map(lambda y:bisect.bisect_right(bins,y),x))
    return quantized

def create_aspect_ratio_groups(dataset,k=0):
    aspect_ratios=_compute_aspect_ratios_custom_dataset(dataset)
    bins=(2**torch.linspace(-1,1,2*k+1)).tolist() if k>0 else [1.0]
    groups=_quantize(aspect_ratios,bins)
    return groups

train_data_loader=torch.utils.data.DataLoader(
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn,
)

def warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor):
    def f(x):
        if x<warmup_iters:
            return 1.0
        a=x/warmup_iters
        return warmup_factor*(1-a)+a
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=f)

def train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=50,warmup=False,scaler=None):
    model.train()
    lr_scheduler=None
    if epoch==0 and warmup is True:
        warmup_factor=1.0/1000
        warmup_iters=1000
        lr_scheduler=warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor)


