import torch
import math

class BoxCoder(object):

    def __init__(self,weights,bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip



class Mather(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    def __init__(self,high_threshhold,low_threshold,allow_low_quality_matches=False):
        self.BETWEEN_THRESHOLDS=-2
        self.BELOW_LOW_THRESHOLD=-1
        assert low_threshold<high_threshhold
        self.high_threshold=high_threshhold
        self.low_threshold=low_threshold
        self.allow_low_quality_matches=allow_low_quality_matches



class BalancedPositiveNegativeSampler(object):
    def __init__(self,batch_size_per_image,positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction