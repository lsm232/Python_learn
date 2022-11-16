from .utils import *

class AssignGTtoDefaultBox(object):
    def __init__(self):
        self.default_box=dboxes300_coco()
        self.encoder=Encoder(self.default_box)
    def __call__(self, image,target):
        boxes=target['boxes']
        labels=target['labels']
        bboxes_out, labels_out = self.encoder.encode(boxes, labels)
        target['boxes'] = bboxes_out
        target['labels'] = labels_out

        return image, target