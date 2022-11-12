import numpy as np

for c in classes:
    dects=det_boxes[c]  #取出预测类别为c的预测框
    gt_class=gt_boxes[c]  #取出标签为c的gt_box
    n_pos=num_pos[c]  #标签为c的gt_box的数量，这里应该等于len(gt_class)

    dects=sorted(dects,key=conf[4],reverse=True)  #按照置信度从高到低对预测框进行排序
    TP=np.zeros(len(dects))   #记录TP的位置
    FP=np.zeros(len(dects))   #记录FP的位置

    for d in range(len(dects)):
        ioumax=0
        if dects[d][-1] in gt_class:
            for j in range(len(gt_class[dects[d][-1]])):
                iou=Evaluator.iou(dects[d][:4],gt_class[dects[d][-1]][:4])
                if iou>ioumax:

                    ioumax=iou
                    jmax=j
                if ioumax>=cfg['threshold']:
                    if gt_class[dects[d][-1]][jmax][-1]==0:
                        TP[d]=1
                    else:
                        FP[d]=1
                else:
                    FP[d]=1
        else:
            FP[d]=1

    acc_FP=np.cumsum(FP)
    acc_TP=np.cumsum(TP)
    rec=acc_TP/n_pos
    prec=np.divide(acc_TP,(acc_TP+acc_FP))


