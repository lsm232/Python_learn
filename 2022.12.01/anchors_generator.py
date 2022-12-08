import torch.nn as nn
import torch
from torch.nn.modules.utils import _pair
import numpy as np

class AnchorGenerator(nn.Module):
    def __init__(self,
                 strides,
                 ratios,
                 scales,
                 base_sizes,  #不同特征图上anchor的大小
                 scale_major,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.
                 ):
        super(AnchorGenerator, self).__init__()
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset'f'!=0, {centers} is given.'
        if not (0<=center_offset<=1):
            raise ValueError('center_offset should be in range(0,1)'f'{center_offset} is given')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'
        self.strides=[_pair(stride) for stride in strides]
        self.base_sizes=[min(stride) for stride in self.strides] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors=self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors=[]
        for i,base_size in enumerate(self.base_sizes):
            center=None
            if self.centers is not None:
                center=self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center,
                )
            )
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,base_size,scales,ratios,center=None):
        w=base_size
        h=base_size
        if center is None:
            x_center=self.center_offset*w
            y_center=self.center_offset*h
        else:
            x_center,y_center=center

        h_ratios=torch.sqrt(ratios)
        w_ratios=1/h_ratios
        if self.scale_major:
            ws=(w*w_ratios[:,None]*scales[None,:]).view(-1)
            hs=(h*h_ratios[:,None]*scales[None,:]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors=[x_center-0.5*ws,y_center-0.5*hs,x_center+0.5*ws,y_center+0.5*hs]
        base_anchors=torch.stack(base_anchors,dim=-1)
        return base_anchors

    def _meshgrid(self,x,y,row_major):
        xx=x.repeat(y.shape[0])
        yy=y.view(-1,1).repeat(1,x.shape[0]).view(-1)
        if not row_major:
            return yy,xx
        else:
            return xx,yy

    def grid_priors(self,featmap_sizes,dtype=torch.float32,device='cuda'):
        assert self.num_levels==len(featmap_sizes)
        multi_level_anchors=[]
        for i in range(self.num_levels):
            anchors=self.single_level_grid_priors(
                featmap_sizes[i],level_idx=i,dtype=dtype,device=device
            )
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,featmap_size,level_idx,dtype=torch.float32,device='cuda'):
        base_anchors=self.base_anchors[level_idx].to(device).to(dtype)
        feat_h,feat_w=featmap_size
        stride_w,stride_h=self.strides[level_idx]
        shift_x=torch.arange(0,feat_w,device=device).to(dtype)*stride_w
        shift_y=torch.arange(0,feat_h,device=device).to(dtype)*stride_h

        shift_xx,shift_yy=self._meshgrid(shift_x,shift_y)
        shifts=torch.stack([shift_xx,shift_yy,shift_xx,shift_yy],dim=-1)
        all_anchors=base_anchors[None,:,:]+shifts[:,None,:]
        all_anchors=all_anchors.view(-1,4)
        return all_anchors

    def sparse_priors(self,prior_idxs,featmap_size,level_idx,dtype=torch.float32,device='cuda'):
        c=1








