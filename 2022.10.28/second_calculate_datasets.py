import os

def calculate_data_txt(txt_path,labels_path):
    assert os.path.exists(labels_path),f"{labels_path} does not exists"
    with open(txt_path,'w') as fid:
        for name in os.listdir(labels_path):
            img_path=os.path.join(labels_path.replace("labels","images"),name.replace(".txt",".jpg"))
            line=img_path+'\n'
            fid.write(line)

def create_data_data(train_txt_path,val_txt_path,classes_info):
    path=r'./data/my_data.data'
    with open(path,'w') as f:
        f.write("classes={}".format(len(classes_info))+ "\n")
        f.write("train={}".format(train_txt_path)+ "\n")
        f.write("val={}".format(val_txt_path)+ "\n")
        f.write("names={}".format('./my_data_label.names')+ "\n")




def main(train_txt_path,train_labels_path,val_txt_path,val_labels_path,classes_label):
    calculate_data_txt(train_txt_path,train_labels_path)
    calculate_data_txt(val_txt_path,val_labels_path)

    classes_info=[line.strip() for line in open(classes_label,'r').readlines() if len(line.strip())>0]
    create_data_data(train_txt_path, val_txt_path, classes_info)






if __name__ == '__main__':
    train_txt_path=r'./data/my_train_data.txt'   #统计数据集后，创建.txt文件，写入每张图像的路径
    train_labels_path=r'F:\my_yolo2\train\labels'

    val_txt_path=r'./data/my_val_data.txt'
    val_labels_path = r'F:\my_yolo2\val\labels'

    classes_label = "./data/my_data_label.names"

    main(train_txt_path,train_labels_path,val_txt_path,val_labels_path,classes_label)