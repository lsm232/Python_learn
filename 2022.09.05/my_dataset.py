from torch.utils.data import Dataset
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import json
import xml.etree.ElementTree as ET
import os
from lxml import etree

Annotation_path=r'G:\leran to play\VOCdevkit\VOC2012\Annotations'
Image_path=r'G:\leran to play\VOCdevkit\VOC2012\JPEGImages'
txt_path=r'./train.txt'





class VocDataste(Dataset):
    def __init__(self,Annotation_root=Annotation_path,Image_root=Image_path,txt_path: str=txt_path,transforms=None):
        assert (os.path.exists(Annotation_root)) and (os.path.exists(Image_root)) and (os.path.exists(txt_path))
        self.Annotation_root = Annotation_root
        self.Image_root = Image_path
        self.txt_path = txt_path

        with open(txt_path) as f:
            xml_files_path=[ Annotation_root+'/'+line.strip()+'.xml' for line in f.readlines() if os.path.exists(Annotation_root+'/'+line.strip()+'.xml')]

        self.xml_list=[]
        for xml_file_path in xml_files_path:
           with open(xml_file_path) as fid:
               xml_str=fid.read()
           xml=etree.fromstring(xml_str)   #将xml转换为element方便处理
           data=self.parse_xml_to_dict(xml)['annotation']
           if 'object' not in data:
               print(f'no objects in {xml_file_path}')
               continue
           self.xml_list.append(xml_file_path)

        json_file=r'pascal_voc_classes.json'
        with open(json_file) as fd:
            self.class_dict=json.load(fd)  #.read会有换行符之类的



        self.transforms=transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        xml=self.xml_list[item]
        with open(xml) as f:
            xml_str=f.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)['annotation']
        img_path=os.path.join(self.Image_root,data["filename"])
        image=Image.open(img_path)
        # plt.imshow(image)
        # plt.show()
        boxes=[]
        labels=[]
        iscrowd=[]
        for obj in data['object']:
            xmin=obj['bndbox']['xmin']
            xmax=obj['bndbox']['xmax']
            ymin=obj['bndbox']['ymin']
            ymax=obj['bndbox']['ymax']

            if xmax<=xmin and ymax<=ymin:
                print(f'文件错误{xml}')
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(obj['difficult'])

            boxes=torch.as_tensor(boxes,dtype=torch.float32)
            labels=torch.as_tensor(labels,dtype=torch.int32)
            iscrowd=torch.as_tensor(iscrowd,dtype=torch.int32)
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

    def parse_xml_to_dict(self, xml):
        if len(xml)==0:
            return {xml.tag:xml.text}
        result={}
        for child in xml:
            child_result=self.parse_xml_to_dict(child)
            if child.tag !='object':
                result[child.tag]=child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag]=[]
                result[child.tag].append(child_result[child.tag])
        return {xml.tag:result}

    def get_height_and_width(self,item):
        xml=self.xml_list[item]
        with open(xml) as fd:
            xml_str=fd.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)['annotation']
        data_height=data['size']["height"]
        data_width=data['size']["width"]
        return data_height,data_width
    










train_data=VocDataste()
data_loader=data.DataLoader(
    dataset=train_data,
    batch_size=2,
    num_workers=0,
    shuffle=False,
    drop_last=True
)

for i,imgs in enumerate(data_loader):
    c=1