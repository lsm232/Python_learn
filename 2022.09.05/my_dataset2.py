import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image
import json
import torch
import os
from torch.utils.data import Dataset
import random
import torch.utils.data as Data
import numpy as np

random.seed(5)


class Object_dataset(Dataset):
    def __init__(self,path=r'G:\leran to play\VOCdevkit\VOC2012',train_ratio=0.5,transforms=None,train_set=True):
        self.annotations_path=os.path.join(path,'Annotations')
        self.img_path=os.path.join(path,'JPEGImages')
        #创建train.txt,test.txt
        self.split_data(train_ratio=train_ratio)
        if train_set:
            with open('train.txt') as f:
                xml_list=[self.annotations_path+'/'+line.strip()+'.xml' for line in f.readlines()]
        else:
            with open('test.txt') as f:
                xml_list=[self.annotations_path+'/'+line.strip()+'.xml' for line in f.readlines()]

        self.xml_list=[]
        for xml_file in xml_list:
            if not os.path.exists(xml_file):
                print(f'{xml_file} not found')
                continue
            with open(xml_file) as f:
                xml=f.read()
            xml_str=etree.fromstring(xml)
            data=self.parse_xml_to_dict(xml_str)['annotation']
            if 'object' not in data:
                print(f'{xml_file} 没有object')
                continue
            self.xml_list.append(xml_file)

        json_file=r'pascal_voc_classes.json'
        with open(json_file) as js:
            self.class_dict=json.load(js)
        self.transforms=transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        xml_path=self.xml_list[item]
        with open(xml_path) as f:
            xml_str=f.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)['annotation']
        img_path=os.path.join(self.img_path,data['filename'])
        image=Image.open(img_path)
        boxes=[]
        labels=[]
        iscrowd=[]
        assert 'object' in data
        for obj in data['object']:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target









    def split_data(self,train_ratio=0.5):
        xml_files=os.listdir(self.annotations_path)
        xml_files_num=len(xml_files)
        train_index=random.sample(range(0,xml_files_num),k=int(xml_files_num*train_ratio))

        train_files=[]
        test_files=[]
        for idx,file in enumerate(xml_files):
            if idx in train_index:
                train_files.append(file.split('.')[0])
            else:
                test_files.append(file.split('.')[0])
        if not os.path.exists('train.txt'):
            tf1=open('train.txt','x')
            tf1.write("\n".join(train_files))
        if not os.path.exists('test.txt'):
            tf2 = open('test.txt','x')
            tf2.write("\n".join(test_files))
        c=1
    def parse_xml_to_dict(self,xml):
        if len(xml)==0:
            return {xml.tag:xml.text}
        result={}
        for child in xml:
            child_result=self.parse_xml_to_dict(child)
            if child.tag !='object' and child.tag !='part':
                result[child.tag]=child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag]=[]
                result[child.tag].append(child_result[child.tag])
        return {xml.tag:result}


train_=Object_dataset()
# train_set=Data.DataLoader(
#     dataset=train_,batch_size=2,shuffle=False,num_workers=0,drop_last=True
# )  每张图片尺寸不一致，会报错

for index in random.sample(range(0,len(train_)),k=4):
    img,tar=train_[index]
    plt.imshow(img)
    plt.show()
    c=1

