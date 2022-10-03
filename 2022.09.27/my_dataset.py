from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from lxml import etree
import json

class VOCDataSet(Dataset):
    def __init__(self,path=r'G:\leran to play\VOCdevkit\VOC2012',transforms=None,txt_name="train.txt"):
        self.img_root = os.path.join(path, "JPEGImages")
        self.annotations_root = os.path.join(path, "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(path, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            xml_list=[os.path.join(self.annotations_root,line.strip()+'.xml') for line in read.readlines() if len(line.strip())>0]

        self.xml_list=[]
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"{xml_path} not found")
                continue

            with open(xml_path) as fid:
                xml_str=fid.read()
            xml=etree.fromstring(xml_str)
            data=self.parse_xml_to_dict(xml)['annotation']
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue

            self.xml_list.append(xml_path)
            json_file=r'./pascal_voc_classes.json'
            with open(json_file) as fid:
                self.class_dict=json.load(fid)
            self.transforms=transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, item):
        xml_path=self.xml_list[item]
        with open(xml_path) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)['annotation']
        img_path=os.path.join(self.img_root,data['filename'])
        img=Image.open(img_path)


        boxes=[]
        labels=[]
        iscrowd=[]
        for obj in data['object']:
            xmin=obj['bndbox']['xmin']
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(self.class_dict[obj['name']])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                image, target = self.transforms(img, target)

            return img, target

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

    def parse_xml_to_dict(self,xml):
        if len(xml)==0:
            return {xml.tag:xml.text}
        result={}
        for child in xml:
            child_result=self.parse_xml_to_dict(child)
            if child.tag!='object':
                result[child.tag]=child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag]=[]
                result[child.tag].append(child_result[child.tag])
        return {xml.tag:result}




data=VOCDataSet()
