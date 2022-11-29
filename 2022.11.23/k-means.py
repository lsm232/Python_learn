import numpy as np


def wh_ious(boxes1,boxes2):
    """
    boxes1:(n,2)
    boxes2:(m,2)
    """
    boxes1=boxes1[:,None]
    boxes2=boxes2[None,:]
    inters=np.minimum(boxes1,boxes2).prod(2)
    ious=inters/(boxes1.prod(2)+boxes2.prod(2)-inters)
    return ious

def kmeans(boxes,k,dist=np.median):
    boxes_num=len(boxes)
    last_clusters=np.zeros(boxes_num)
    clusters=boxes[np.random.choice(boxes,size=k,replace=False)]
    while True:
        distance=[]
        for i in range(boxes_num):
            ious=wh_ious(boxes[i],clusters)
            dis=[1-iou for iou in ious]
            distance.append(dis)
        distance=np.array(distance)
        new_clusters=np.argmin(distance,axis=1)
        if (new_clusters==last_clusters).all():
            break
        for j in range(k):
            clusters[j]=dist(boxes[j==new_clusters],axis=0)
        last_clusters=new_clusters
    return clusters



