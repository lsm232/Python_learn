import torch
import torch.nn as nn

class GeneralizedRCNNTransform(nn.Module):
    def __init__(self,min_size,max_size,image_mean,image_std):
        super(GeneralizedRCNNTransform, self).__init__()

        self.min_size=min_size
        self.max_size = max_size  # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std