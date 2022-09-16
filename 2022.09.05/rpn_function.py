import torch
from typing import List,Tuple,Dict
from torch import Tensor
import torch.nn as nn

class AnchorGenerator(nn.Module):
    def __init__(self,sizes,aspect_ratios):
        super(AnchorGenerator, self).__init__()
        self.sizes=sizes   #((64,),(128,),(256,))
        self.aspect_ratios=aspect_ratios   #(0.5,1,2)
        self._cache={}

    def gird_anchors(self,grid_sizes,strides):
        #type: (List[List[int]],List[List[Tensor]]) -> List[Tensor]
        shifts_=[]
        for grid_sizes,strides in zip(grid_sizes,strides):
            f_h,f_w=grid_sizes
            s_h,s_w=strides
            hs=torch.arange(0,f_h)*s_h   #torch.range 返回float且包括end，arange返回int,不包括end
            ws=torch.arange(0,f_w)*s_w
            shift_y,shift_x=torch.meshgrid(hs,ws)
            shift_y=shift_y.reshape(-1)
            shift_x=shift_x.reshape(-1)
            shifts=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1)
            shifts_.append(shifts)
        return shifts_   #(n,4)

    def generate_anchors(self,sizes,aspect_raios):
        sizes=torch.as_tensor(sizes)
        aspect_raios=torch.as_tensor(aspect_raios)
        h_ratios=torch.sqrt(aspect_raios)
        w_ratios=1/h_ratios
        ws=(w_ratios[:,None]*sizes[None,:]).view(-1)
        hs=(h_ratios[:,None]*sizes[None,:]).view(-1)
        base_anchors=torch.stack([-ws,-hs,ws,hs],dim=1)/2
        return base_anchors.round()  #(3,4)


    def set_cell_anchors(self):
        cell_anchors=[
            self.generate_anchors(sizes,aspect_ratios)
            for sizes,aspect_ratios in zip(self.sizes,self.aspect_ratios)
        ]
        self.cell_anchors=torch.cat(cell_anchors,dim=0)

    def cached_gird_anchors(self,gird_sizes,strides):
        # type: (List[List[int]],List[List[Tensor]]) -> List[Tensor]
        key=str(gird_sizes)+str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors=self.grid_anchors(gird_sizes,strides)
        self._cache[key]=anchors
        return anchors


    def forward(self,imagelist,feature_maps):
        #求出feature map与原图像的比，用于移动anchors
        grid_sizes=list([feature_map.shape[-2:] for feature_map in feature_maps])  #feature_map.shape[-2:]=torch.Size([200,500]),torch.tensor(torch.Size([200,500]))=tensor([200,500])
        pad_sizes=imagelist.batch_imgs.shape[-2:]
        strides=[[torch.tensor(pad_sizes[0]//g[0]),torch.tensor(pad_sizes[1]//g[1])] for g in grid_sizes]  #torch.tensor不对数据类型进行转换，torch.Tensor是torch.FloatTensor的缩写，会将数据转换为浮点型

        #计算偏移量
        shifts=self.cached_gird_anchors(grid_sizes,strides)

        #生成anchors
        self.set_cell_anchors()

        #将偏移量与anchors相加，相当于移动anchors
        shifts_anchors=shifts.view(-1,1,4)+self.cell_anchors.view(1,-1,4)
        shifts_anchors=shifts_anchors.reshape(-1,4)

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg










