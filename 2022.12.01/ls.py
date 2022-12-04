def nms(boxes,scores,thresh=0.5):
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]

    _,order=scores.sort(0,descending=True)
    keep=[]
    while order.numel()>0:
        if order.numel()==1:
            keep.append(order[0])
            break
        else:
            keep.append(order[0])
            xx1=x1[order[1:]].clamp(min=x1[0])
            yy1=y1[order[1:]].clamp(min=y1[0])
            xx2 = x2[order[1:]].clamp(max=x2[0])
            yy2 = y2[order[1:]].clamp(max=y2[0])

            inter=(yy2-yy1).clamp(min=0)*(xx2-xx1).clamp(min=0)
            ious=inter/(x1*y1+x2*y2-inter)
            idx=(ious<=thresh).nonzero().squeeze()

            if idx.numel()==0:
                break
            else:
                order=order[idx+1]
    return keep




