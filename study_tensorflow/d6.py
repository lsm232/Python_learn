import numpy as np

cfg={'threshold':0.6}
det_boxes={'class1':[[],[]],'class2':[[],[]]}
gt_boxes={'class1':{'name1':[[],[]],'name2':[[],[]]},'class2':{'name3':[[],[]],'name4':[[],[]]}}
classes=['class1','class2']

def cal_iou(box1,box2):
    c=1
    return c


for cls in classes:
    dects=det_boxes[cls] #[[left,top,right,bottom,score,name],[],...]
    gt_class=gt_boxes[cls] #{'name1':[[left,top,right,bottom,flag],[],...]}
    num_pos=len(gt_class)
    tp=np.zeros(len(dects))
    fp=np.zeros(len(dects))



    sorted_dects=sorted(dects,key=lambda conba:conba[-2],reverse=True)

    for d in sorted_dects:
        iou_max = 0.
        if d[-1] in gt_class:
            for g in gt_class[d[-1]]:
                i = 0
                iou=cal_iou(d[:4],g[:4])
                if iou>iou_max:
                    iou_max=iou
                    iou_max_index=i
            if iou_max>cfg['threshold'] and g[-1]!=1:
                tp[i]=1
                g[-1]=1
            else:
                fp[i]=1
        else:
            fp[i]=1
        i+=1
    acc_fp=np.cumsum(fp)
    acc_tp=np.cumsum(tp)
    recall=acc_tp/num_pos
    prec=np.divide(acc_tp,(acc_tp+acc_fp))







