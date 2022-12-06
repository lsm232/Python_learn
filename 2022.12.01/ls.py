import numpy as np
import torch.nn.functional as F
import torch
import math
import itertools
import torch.nn as nn


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def calc_iou_tensor(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


class Encoder(object):
    def __init__(self, dboxes):
        self.dboxes = dboxes(order='ltrb')
        self.dboxes_xywh = dboxes(order='xywh').unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int32)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.int32)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        x = 0.5 * (bboxes_out[:, 0] + bboxes_out[:, 2])  # x
        y = 0.5 * (bboxes_out[:, 1] + bboxes_out[:, 3])  # y
        w = bboxes_out[:, 2] - bboxes_out[:, 0]  # w
        h = bboxes_out[:, 3] - bboxes_out[:, 1]  # h
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out


class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5
        self.max_output = 100

    def scale_back_batch(self, bboxes_in, scores_in):
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]  # 预测的x, y回归参数
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]  # 预测的w, h回归参数

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, :2] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 0]
        t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 0]
        b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l  # xmin
        bboxes_in[:, :, 1] = t  # ymin
        bboxes_in[:, :, 2] = r  # xmax
        bboxes_in[:, :, 3] = b  # ymax

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
        device = bboxes_in.device
        num_classes = scores_in.shape[-1]

        bboxes_in = bboxes_in.clamp(min=0, max=1)
        bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores_in)

        bboxes_in = bboxes_in[:, 1:, :]
        scores_in = scores_in[:, 1:]
        labels = labels[:, 1:]

        bboxes_in = bboxes_in.reshape(-1, 4)
        scores_in = scores_in.reshape(-1)
        labels = labels.reshape(-1)

        inds = torch.where(torch.gt(scores_in, 0.05))[0]
        bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]

        ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
        keep = (ws >= 1 / 300) & (hs >= 1 / 300)
        keep = torch.where(keep)[0]
        bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

        keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)
        keep = keep[:num_output]
        bboxes_out = bboxes_in[keep, :]
        scores_out = scores_in[keep]
        labels_out = labels[keep]

        return bboxes_out, labels_out, scores_out

    def forward(self, bboxes_in, scores_in):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        outputs = []

        for bbox, prob in zip(bboxes.split(0, 1), probs.split(0, 1)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
        return outputs


def batched_nms(boxes, scores, idxs, iou_threshold):
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]  # 咋算的
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size
        self.feat_size = feat_size
        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh
        self.steps = steps
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        fk = fig_size / np.array(steps)

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = math.sqrt(sk1 * sk2)

            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)
        self.dboxes.clamp_(min=0, max=1)

        self.dboxes_ltrb = self.dboxes.clone()
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






