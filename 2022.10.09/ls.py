from torch.utils.data import Dataset
import torch
import json
import os
from lxml import etree
from PIL import Image
import random
from torchvision.transforms import functional as F



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
data_transform={
    "train":Compose([ToTensor(),RandomHorizontalFlip(0.5)]),
    "val":Compose([ToTensor()]),
}
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

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

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





my_data=vocDataset()
h,w=my_data[1].get_height_and_width
x=1