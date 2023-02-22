
def iou(boxA,boxB):
    left_max=max(boxA[0],boxB[0])
    top_max=max(boxA[1],boxB[1])
    right_min=min(boxA[2],boxB[2])
    bottom_min=min(boxA[3],boxB[3])
    inter=max(0,bottom_min-top_max)*max(0,right_min-left_max)
    area_B=(boxB[3]-boxB[1])*(boxB[2]-boxB[0])
    area_A=(boxA[3]-boxA[1])*(boxA[2]-boxA[0])
    iou=inter/(area_A+area_B-inter)
    return iou
