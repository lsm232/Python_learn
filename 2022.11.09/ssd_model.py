from .res50_backbone import *
import torch
import torch.nn as nn
from .utils import *

class Backbone(nn.Module):
    def __init__(self,pretrain_path):
        super(Backbone, self).__init__()
        net=resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))
        self.feature_extractor=nn.Sequential(*list(net.children())[:7])

        conv4_block1=self.feature_extractor[-1][0]

        conv4_block1.conv1.stride=1
        conv4_block1.conv2.stride=1
        conv4_block1.downsample[0].stride=1

    def forward(self,x):
        x=self.feature_extractor(x)
        return x

class SSD300(nn.Module):
    def __init__(self,backbone,num_classes):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone,"out_channels"):
            raise Exception("backbone have not attribute:out_channels")

        self.feature_extractor=backbone
        self.num_classes=num_classes
        self._build_additional_layers(backbone.out_channels)

        self.num_defaults=[4,6,6,6,4,4]
        location_extractors=[]
        confidence_extractors=[]

        for nd,oc in zip(self.num_defaults,self.feature_extractor.out_channels):
            location_extractors.append(nn.Conv2d(oc,nd*4,kernel_size=3,padding=1))
            confidence_extractors.append(nn.Conv2d(oc,nd*num_classes,kernel_size=3,padding=1))

        self.loc=nn.ModuleList(location_extractors)
        self.conf=nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_box=dboxes300_coco()
        self.compute_loss=Loss(default_box)



    def __build_additional_layers(self,channels):
        middle_channels=[256,256,128,128,128]
        additional_blocks=[]

        for i,(input_nc,middle_nc,output_nc) in enumerate(zip(channels[:-1],middle_channels,channels[1:])):
            padding,stride=(1,2) if i<3 else (0,1)
            layer=nn.Sequential(
                nn.Conv2d(input_nc,middle_nc,1,bias=False),
                nn.BatchNorm2d(middle_nc),
                nn.ReLU(True),
                nn.Conv2d(middle_nc,output_nc,kernel_size=3,stride=stride,padding=padding,bias=False),
                nn.BatchNorm2d(output_nc),
                nn.ReLU(True),
            )
            additional_blocks.append(layer)
            self.additional_blocks=nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers=[*self.additional_blocks,*self.loc,*self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim()>1:
                    nn.init.xavier_normal_(param)

class Loss(nn.Module):
    def __init__(self,dboxes):
        super(Loss, self).__init__()
        self.scale_xy=1.0/dboxes.scale_xy
        self.scale_wh=1.0/dboxes.scale_wh

        self.location_loss=nn.SmoothL1Loss(reduction='none')
        self.dboxes=nn.Parameter(dboxes(order='xywh').transpose(0,1).unsqueeze(dim=0),requires_grad=False)

        self.confidence_loss=nn.CrossEntropyLoss(reduction='none')

    def _location_vec(self,loc):
        gxy=self.scale_xy*(loc[:,:2,:]-self.dboxes[:,:2,:])/self.dboxes[:,2:,:]
        gwh=self.scale_wh*(loc[:,2:,:]/self.dboxes[:,2:,:]).log()
        return torch.cat([gxy,gwh],dim=1).contiguous()

    def forward(self,ploc,plabel,gloc,glabel):
        mask=torch.gt(glabel,0)
        pos_num=mask.sum(dim=1)
        vec_gd=self._location_vec(gloc)

        loc_loss=self.location_loss(ploc,vec_gd).sum(dim=1)
        loc_loss=(mask.float()*loc_loss).sum(dim=1)

        con = self.confidence_loss(plabel, glabel)
        con_neg=con.clone()
        con_neg[mask]=0.0
        _,con_idx=con_neg.sort(dim=1,descending=True)
        _,con_rank=con_idx.sort(dim=1)

        neg_num=torch.clamp(3*pos_num,max=mask.size(1)).unsqueeze(-1)


        #今天真倒霉


