"""Encode object boxes and labels"""
import os, sys
import math
import torch
from .utils import *



class DataEncoder(object):
    def __init__(self):
        self.anchor_areas = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 512 * 512]  # p3->p7
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        """
        Compute anchor width and height for each feature map.
        :return: anchor_wh: (tensor) anchor wh, (size) [#fm, #anchors_per_cell, 2]
        """
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h=ar
                w = math.sqrt(s * ar)
                h = w / ar
                for sr in self.scale_ratios:
                    anchor_w = w * sr
                    anchor_h = h * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        """
        Compute anchor boxes for each feature map.
        :param input_size: the size of input image
        :return: boxes: (list) anchor boxes for each feature map. Each of size [#anchors, 4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]
        # num_anchors_per_level = [int(fs[0]) * int(fs[1]) * 9 for fs in fm_sizes]

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5  # plus 0.5 to each cell's center, [fm_w*fm_h, 2]
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(
                fm_h, fm_w, 9, 2)  # convert the center to original image, and expand to all anchors
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat([xy, wh], 3)  # [x, y, w, h]
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        """
        Encode target bounding boxes and class labels.
        we obey the Faster RCNN box coder:
        tx = (x - anchor_x) / anchor_w
        ty = (y - anchor_y) / anchor_h
        tw = log(w / anchor_w)
        th = log(h / anchor_h)

        :param boxes: (tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [#obj, 4].
        :param labels: (tensor) object class labels, sized [#obj,].
        :param input_size: (int/tuple) input size of the original image
        :return:
            loc_targets: (tensor) encoded bounding boxes, sized [#anchors, 4].
            cls_targets: (tensor) encoded class labels, sized [#anchors,].
        """
        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = find_jaccard_overlap(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        loc_targets = loc_targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
        cls_targets = labels[max_ids]

        cls_targets[max_ious < 0.4] = 0
        ignore = (max_ious >= 0.4) & (max_ious < 0.5)  # ignore ious between [0:q.4, 0.5]
        cls_targets[ignore] = -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        """
        Decode outputs back to bounding box locations and class labels.
        :param loc_preds: (tensor) predicted locations, sized [#anchors, 4]
        :param cls_preds: (tensor) predicted class labels, sized [#anchors, #classes]
        :param input_size: (int/tuple) the input size of original image
        :return:
            boxes: (tensor) decode box locations, sized [#obj, 4]
            labels: (tensor) class labels for each box, sized [#obj,].
        """
        CLS_THRESH = 0.05
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size).cuda()
        loc_preds = loc_preds * torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]
        #print(loc_xy.shape, 'loc_xy')
        #print(anchor_boxes[:, 2:].shape, 'anchor boxes')
        #print(anchor_boxes[:, :2].shape, 'anchor boxes')
        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)
        # boxes = torch.cat([xy, wh], 1)
        # boxes = change_box_order(boxes, 'xywh2xyxy')
        # boxes[:, 0:3:2] = boxes[:, 0:3:2].clamp(0, input_size[0])
        # boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, input_size[1])

        # import pdb; pdb.set_trace()
        scores, labels = cls_preds.sigmoid().max(1)
        ids = scores > CLS_THRESH
        ids = ids.nonzero().squeeze()
        # print(ids, 'ids')
        
        new_boxes = boxes.clone()
        new_scores = scores.clone()
     
        keep = box_nms(new_boxes, new_scores, NMS_THRESH)
        # print(keep, 'keep')
        # keep = keep.cuda()
        return boxes[ids][keep], labels[ids][keep] + 1, scores[ids][keep]