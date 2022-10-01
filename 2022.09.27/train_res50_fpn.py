from .resnet50_fpn_model import *
import torch


def create_model(num_classes,load_pretrain_weights=False):
    backbone=resnet50_fpn_backbone(pretrain_path='',norm_layer=torch.nn.BatchNorm2d,layer_to_train=3)
