import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

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






