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

    def gen_single_level_base_anchors(self,base_sizes,scales,ratios,center=None):
        




