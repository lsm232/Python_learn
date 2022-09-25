import torch
import torch.nn as nn
from typing import List
from torchvision.ops import roi_align

class MultiScaleRoIAlign(nn.Module):
    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        self.feature_names=featmap_names
        self.scales = None

        self.map_levels = None

    def convert_to_roi_format(self,boxes):
        concat_boxes=torch.cat(boxes,0)
        ids=torch.cat([torch.full_like(b[:,:1],i) for i,b in enumerate(boxes)],dim=0)
        rois=torch.cat([ids,concat_boxes],dim=1)
        return rois

    def inter_scale(self,feature,original_size):
        size = feature.shape[-2:]
        possible_scales = torch.jit.annotate(List[float], [])
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]

    def setup_scales(self,features,image_shapes):
        max_x=0
        max_y=0
        for image_shape in image_shapes:
            max_x=max(max_x,image_shape[0])
            max_y=max(max_y,image_shape[1])
        original_input_shape = (max_x, max_y)
        scales=[self.inter_scale(feat,original_input_shape) for feat in features]

        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.scales = scales
        self.map_levels = initLevelMapper(int(lvl_min), int(lvl_max))




    def forward(self,features_per_level,boxes,image_shapes):
        x_filtered=[]
        for k,v in features_per_level.items():
            if k in self.feature_names:
                x_filtered.append(v)
        num_levels=len(x_filtered)
        rois=self.convert_to_roi_format(boxes)
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)
        scales = self.scales
        assert scales is not None

        if num_levels == 1:
            return roi_align(
                x_filtered[0], rois,
                output_size=self.output_size,
                spatial_scale=scales[0],
                sampling_ratio=self.sampling_ratio
            )
        mapper=self.map_levels
        levels=mapper(boxes)




