import os
import json
import shutil
from tqdm import tqdm
from lxml import etree


def parse_xml_to_dict(xml):
    if len(xml)==0:
        return {xml.tag: xml.text}
    results={}
    for child in xml:
        child_results=parse_xml_to_dict(child)
        if child.tag!='object':
            results[child.tag]=child_results[child.tag]
        else:
            if 'object' not in results:
                results['object']=[]

            results['object'].append(child_results[child.tag])
    return {xml.tag:results}


def translate_info(voc_images_path,voc_labels_path,files_name,save_file_root,class_dict,train_val):
    #创建路径为...../train or val/labels
    save_labels_path=os.path.join(save_file_root,train_val,'labels')
    if os.path.exists(save_labels_path) is False:
        os.makedirs(save_labels_path)

    save_images_path=os.path.join(save_file_root,train_val,'images')
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file_name in tqdm(files_name,desc="translate {} file...".format(train_val)):
        img_path=os.path.join(voc_images_path,file_name+'.jpg')
        assert os.path.exists(img_path),f"{img_path} dose not exit"

        label_path=os.path.join(voc_labels_path,file_name+'.xml')
        assert os.path.exists(label_path), f"{label_path} dose not exit"

        with open(label_path) as f:
            xml_str=f.read()
        xml=etree.fromstring(xml_str)
        data=parse_xml_to_dict(xml)['annotation']
        img_height=int(data['size']['height'])
        img_width=int(data['size']['width'])

        assert 'object' in data.keys(),f'{label_path} have no object'
        if len(data['object'])==0:
            print('the file {} have no object,skip'.format(img_path))
            continue

        txt_path=os.path.join(save_labels_path,file_name+'.txt')
        with open(txt_path,'w') as f:
            for index,obj in enumerate(data['object']):
                xmin=float(obj['bndbox']['xmin'])
                xmax=float(obj['bndbox']['xmax'])
                ymin = float(obj['bndbox']['ymin'])
                ymax = float(obj['bndbox']['ymax'])

                center_x=xmin+(xmax-xmin)/2
                center_y=ymin+(ymax-ymin)/2
                width=xmax-xmin
                height=ymax-ymin

                center_x=round(center_x/img_width,6)
                center_y=round(center_y/img_height,6)
                width=round(width/img_width,6)
                height=round(height/img_height,6)

                if xmin>=xmax or ymin>ymax:
                    print("{} have problem".format(label_path))

                class_name=obj['name']
                class_index=class_dict[class_name]-1  #为什么要-1

                info=[str(i) for i in [class_index,center_x,center_y,width,height]]

                if index==0:
                    f.write(" ".join(info))
                else:
                    f.write("\n"+" ".join(info))

        path_copy_to=os.path.join(save_images_path,img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path,path_copy_to)


def create_class_name(class_dict):
    keys=class_dict.keys()
    with open('./data/my_data_label.names',"w") as w:
        for i,key in enumerate(keys):
            if i+1==len(keys):
                w.write(key)
            else:
                w.write(key+"\n")






def main(voc_path,label_json_path,save_file_path):
    voc_images_path=os.path.join(voc_path,"JPEGImages")
    voc_labels_path=os.path.join(voc_path,"Annotations")
    voc_train_txt_path=os.path.join(voc_path,"ImageSets", "Main","train.txt")
    voc_val_txt_path=os.path.join(voc_path,"ImageSets", "Main","val.txt")

    assert os.path.join(label_json_path),"label_json_path does not exist..."
    assert os.path.join(voc_images_path),"voc images path does not exist..."
    assert os.path.join(voc_labels_path),"voc labels path does not exist..."
    assert os.path.join(voc_train_txt_path),"voc train.txt path does not exist..."
    assert os.path.join(voc_val_txt_path),"voc val.txt path does not exist..."

    if os.path.exists(save_file_path) is False:
        os.makedirs(save_file_path)

    json_file=open(label_json_path,'r')
    class_dict=json.load(json_file)

    with open(voc_train_txt_path) as fid:
        train_file_names=[i for i in fid.read().splitlines() if len(i.strip())>0]
    translate_info(voc_images_path,voc_labels_path,train_file_names,save_file_path,class_dict,'train')

    with open(voc_val_txt_path) as fid:
        val_file_names=[i for i in fid.read().splitlines() if len(i.strip())>0]
    translate_info(voc_images_path,voc_labels_path,val_file_names,save_file_path,class_dict,'val')

    create_class_name(class_dict)






if __name__=='__main__':

    voc_path=r'G:\leran to play\VOCdevkit\VOC2012'
    label_json_path=r'./data/pascal_voc_classes.json'
    save_file_path=r'F:\my_yolo2'

    main(voc_path, label_json_path,save_file_path)