import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler
from collections import defaultdict


def compute_aspect_ratios(dataset):
    indices=range(len(dataset))
    aspect_ratios=[]
    for i in indices:
        height,width=dataset.get_height_and_width(i)
        aspect_ratio=float(height/width)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    # bisect_right：寻找y元素按顺序应该排在bins中哪个元素的右边，返回的是索引
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(dataset,k=0):
    aspect_ratios=compute_aspect_ratios(dataset)
    bins=(2**np.linspace(-1,1,2*k+1)).tolist() if k>0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))
    return groups

class GroupedBatchSampler(BatchSampler):
    def __init__(self,sampler,group_ids,batch_size):
        self.sampler=sampler
        self.group_ids=group_ids
        self.batch_size=batch_size
    def __iter__(self):
        buffer_per_group=defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches=0
        for idx in self.sampler:

