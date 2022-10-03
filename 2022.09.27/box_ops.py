import torch

def box_area(box):
    width=box[:,2]-box[:,0]
    height=box[:,3]-box[:,1]
    return width*height



def box_iou(boxes1,boxes2):
    boxes1_area=box_area(boxes1)
    boxes2_area=box_area(boxes2)
    lt=torch.max(boxes1[:,None,:2],boxes2[:,:2])
    rb=torch.min(boxes1[:,None,2:],boxes2[:,2:])

    wh=(rb-lt).clamp(min=0)
    inter=wh[:,:,0]*wh[:,:,1]
    iou=inter/(boxes1_area[:,None]+boxes2_area-inter)
    return iou