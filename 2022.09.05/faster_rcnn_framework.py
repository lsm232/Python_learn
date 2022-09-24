import torch
import torch.nn as nn
from collections import OrderedDict

class FasterRcnnBase(nn.Module):
    def __init__(self,backbone,rpn,roi_heads,transform):
        super(FasterRcnnBase, self).__init__()
        self.backbone=backbone
        self.rpn=rpn
        self.roi_heads=roi_heads
        self.transform=transform

    def forward(self,images,targets):
        if self.training and targets is None:
            raise ValueError("训练，targets不应该为空")
        original_images_size=[]
        for img in images:
            original_images_size.append((img.shape[-2],img.shape[-1]))
        images_list,targets=self.transform(images,targets)
        features=self.backbone(images_list.tensor)
        if isinstance(features,torch.Tensor):
            features=OrderedDict(['0',features])
        proposals,proposal_losses=self.rpn(images_list,features,targets)
        detections,detector_losses=self.roi_heads(features,proposals,images_list.image_sizes,original_images_size)






