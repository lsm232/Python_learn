import torch.nn as nn
from typing import Tuple,List,Dict,Tuple
from torch import Tensor
from transform import Imagelist
import torch

class AnchorGenerator(nn.Module):
    def __init__(self,sizes,aspect_ratios):
        super(AnchorGenerator, self).__init__()
        self.sizes=sizes
        self.aspect_ratios=aspect_ratios


    def gird_anchors(self,grid_sizes,strides):
        anchors=[]
        cell_anchors=self.cell_anchors

        for size,stride,base_anchors in zip(grid_sizes,strides,cell_anchors):
            feature_height,feature_width=size
            stride_height,stride_width=strides
            shifts_y=torch.arange(0,feature_height)*stride_height
            shifts_x=torch.arange(0,feature_width)*stride_width
            shift_y,shift_x=torch.meshgrid(shifts_y,shifts_x)
            shift_x=shift_x.reshape(-1)
            shift_y=shift_y.reshape(-1)
            shifts=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]


    def generate_anchors(self,sizes,aspect_ratios):
        #h_ratios/w_ratios=aspect_ratios   1=h_r*w_r
        w_ratios=torch.as_tensor(torch.sqrt(aspect_ratios))
        h_ratios=torch.as_tensor(1/w_ratios)
        ws=(w_ratios[None,:]*sizes[:,None]).view(-1)
        hs=(h_ratios[None,:]*sizes[:,None]).view(-1)
        base_anchors=[-ws,-hs,ws,hs]/2
        base_anchors=torch.stack(base_anchors,dim=1)
        return base_anchors.round()

    def cached_gird_anchors(self,grid_sizes,strides):
        key=str(grid_sizes)+str(strides)
        anchors=self.grid_anchors(grid_sizes,strides)
        self._cache[key]=anchors
        return anchors





    def set_cell_anchors(self):
        self.cell_anchors=[self.generate_anchors(sizes,aspect_ratios) for sizes,aspect_ratios in zip(self.sizes,self.aspect_ratios)]



    def forward(self,image_list,feature_maps):
        # type: (Imagelist,List[Tensor])->List[Tensor]
        grid_sizes=list([feature_map.shape[-2:]] for feature_map in feature_maps)
        pad_image_sizes=image_list.batch_imgs.shape[-2:]
        strides=[[torch.tensor(pad_image_sizes[0]/g[0]),torch.tensor(pad_image_sizes[1]/g[1])] for g in grid_sizes]
        self.sel_cell_anchors()
        anchors_over_all_features=self.cached_grid_anchors(grid_sizes,strides)
        anchors=[]
        for i,(image_height,image_width) in enumerate(image_list.resized_sizes):
            anchors_in_image=[]
            for anchors_per_feature_map in anchors_over_all_features:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors

