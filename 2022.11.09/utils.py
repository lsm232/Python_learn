import numpy as np
import torch
import math
import itertools


def dboxes300_coco():
    figsize=300
    feat_size=[38,19,10,5,3,1]
    steps=[8,16,32,64,100,300]  #咋算的
    scales=[21,45,99,153,207,261,315]
    aspect_ratios=[[2],[2,3],[2,3],[2,3],[2],[2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

class DefaultBoxes(object):
    def __init__(self,fig_size,feat_size,steps,scales,aspect_ratios,scale_xy=0.1,scale_wh=0.2):
        self.fig_size=fig_size
        self.feat_size=feat_size
        self.scale_xy_=scale_xy
        self.scale_wh_=scale_wh
        self.steps=steps
        self.scales=scales
        self.aspect_ratios=aspect_ratios

        fk=fig_size/np.array(steps)

        self.default_boxes=[]
        for idx,sfeat in enumerate(self.feat_size):
            sk1=scales[idx]/fig_size
            sk2=scales[idx+1]/fig_size
            sk3=math.sqrt(sk1*sk2)

            all_sizes=[(sk1,sk1),(sk3,sk3)]

            for alpha in aspect_ratios[idx]:
                w,h=sk1*math.sqrt(alpha),sk1/math.sqrt(alpha)
                all_sizes.append((w,h))
                all_sizes.append((h,w))

            for w,h in all_sizes:
                for i,j in itertools.product(range(sfeat),repeat=2):
                    cx,cy=(j+0.5)/fk[idx],(i+0.5)/fk[idx]
                    self.default_boxes.append((cx,cy,w,h))


        self.dboxes=torch.as_tensor(self.default_boxes,dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1)

        self.dboxes_ltrb=self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]  # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]  # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]  # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]  # ymax

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order='ltrb'):
        # 根据需求返回对应格式的default box
        if order == 'ltrb':
            return self.dboxes_ltrb

        if order == 'xywh':
            return self.dboxes






