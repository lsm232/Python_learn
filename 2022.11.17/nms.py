def nms(bboxes,scores,thresh=0.5):
    x1=bboxes[:,0]
    y1=bboxes[:,1]
    x2=bboxes[:,2]
    y2=bboxes[:,3]
    areas=(x2-x1)*(y2-y1)

    _,order=scores.sort(0,descending=True)
    
