import os
import random

#给定标注文件所在路径，将.xml文件划分为训练集和测试集

annotations_xml_path=r'G:\leran to play\VOCdevkit\VOC2012\Annotations'
train_ratio=0.5    #训练集占比

assert os.path.exists(annotations_xml_path),"path {} 不存在！！！！".format(annotations_xml_path)
xml_files=os.listdir(annotations_xml_path)
xml_files_name=[]
for xml_file in xml_files:
    xml_file_name=xml_file.split('.')[0]
    xml_files_name.append(xml_file_name)

len=len(xml_files_name)
train_index=random.sample(range(0,len),k=int(len*train_ratio))     #range(0,len)范围为0-(len-1)

train_files_name=[]
test_files_name=[]
for index,file_name in enumerate(xml_files_name):
    if index in train_index:
        train_files_name.append(file_name)
    else:
        test_files_name.append(file_name)


train_f=open("train.txt","x")
test_f=open(r'test.txt','x')
train_f.write("\n".join(train_files_name))
test_f.write("\n".join(test_files_name))


