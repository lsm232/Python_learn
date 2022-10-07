import torch

def clip_boxes_to_image(boxes,size):
    dim=boxes.dim()
    boxes_x=boxes[...,0::2]
    boxes_y=boxes[...,1::2]
    height,width=size

    boxes_x=torch.clamp(min=0,max=width)
    boxes_y=torch.clamp(min=0,max=height)
    clipped_boxes=torch.stack([boxes_x,boxes_y],dim=dim)
    return clipped_boxes.reshape(boxes.shape)

def remove_small_boxes(boxes,min_size):
    ws,hs=boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]
    keep=torch.logical_and(torch.ge(ws,min_size),torch.ge(hs,min_size))
    keep = torch.where(keep)[0]
    return keep

def batches_nms(boxes,scores,idxs,iou_threshhold):
    max_coordinate=boxes.max()
    offsets=idxs*(max_coordinate+1)
    boxes_for_nms=boxes+offsets[:,None]
    keep=nms(boxes_for_nms,scores,iou_threshhold)
    return keep