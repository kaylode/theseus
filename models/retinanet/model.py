import os, sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from .fpn import getFPN
from .utils import *


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=80, backbone = 'FPN50', pretrained = False, input_size = (300,300), device = None):
        super(RetinaNet, self).__init__()
        
        self.device = device if device is not None else torch.device("cpu")
        self.anchor_areas = [32 * 32, 64 * 64, 128 * 128, 256 * 256, 512 * 512]  # p3->p7
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh()
        self.input_size = input_size

        self.fpn = getFPN(backbone)
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        
        self.priors_xywh = self._get_anchor_boxes(torch.Tensor(input_size))
        self.priors_xy = change_box_order(self.priors_xywh,'xywh2xyxy')
        self.priors_cxcy = change_box_order(self.priors_xy,'xyxy2cxcy')
        
        if self.pretrained:
            self.load_state_dict(backbone)
        self.freeze_bn()
        
    def forward(self, inputs):
        x = inputs
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        prediction = {}
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4, H, W] -> [N, H, W, 9*4] -> [N, H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N, 9*80, H, W] -> [N, H, W, 9*80] -> [N, H*W*9, 80]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        
        return loc_preds, cls_preds
    
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

            # Normalize coordinate
            i_h, i_w = input_size
            box[:,:,:,0] = box[:,:,:,0]*1.0 / i_w
            box[:,:,:,1] = box[:,:,:,1]*1.0 / i_h
            box[:,:,:,2] = box[:,:,:,2]*1.0 / i_w
            box[:,:,:,3] = box[:,:,:,3]*1.0 / i_h

            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0).to(self.device)
    

    def detect(self, loc_preds, cls_preds, min_score=0.01, nms_thresh = 0.5):
        """
        Decode outputs back to bounding box locations and class labels.
        :param loc_preds: (tensor) predicted locations, sized [#anchors, 4]
        :param cls_preds: (tensor) predicted class labels, sized [#anchors, #classes]
        :param input_size: (int/tuple) the input size of original image
        :return:
            boxes: (tensor) decode box locations, sized [#obj, 4]
            labels: (tensor) class labels for each box, sized [#obj,].
        """

        
        batch_size = loc_preds.shape[0] # N
        input_size = torch.Tensor(self.input_size)

        anchor_boxes = self._get_anchor_boxes(input_size)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        anchor_boxes = torch.cat([anchor_boxes for i in range(batch_size)], dim = 0) # [N, #anchors, 4]
        
        #loc_preds = loc_preds * torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(self.device)
        loc_xy = loc_preds[:, :, :2]
        loc_wh = loc_preds[:, :, 2:]
        #print(loc_xy.shape, 'loc_xy')
        #print(anchor_boxes[:, 2:].shape, 'anchor boxes')
        #print(anchor_boxes[:, :2].shape, 'anchor boxes')
        xy = loc_xy * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2]
        wh = loc_wh.exp() * anchor_boxes[:, :, 2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], dim = 2) # [N, #anchors, 4]
        # boxes = torch.cat([xy, wh], 1)
        # boxes = change_box_order(boxes, 'xywh2xyxy')
        # boxes[:, 0:3:2] = boxes[:, 0:3:2].clamp(0, input_size[0])
        # boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, input_size[1])
        
        # import pdb; pdb.set_trace()
        scores, labels = cls_preds.sigmoid().max(2)
        ids = scores > min_score
        ids = ids.nonzero().squeeze()
        print(ids, 'ids')
        print(scores)
        new_boxes = boxes.clone()
        new_scores = scores.clone()
     
        keep = box_nms(new_boxes, new_scores, nms_thresh)
        print(keep, 'keep')
        # keep = keep.cuda()
        return {
            'boxes':boxes[ids][keep],
            'labels': labels[ids][keep] + 1,
            'scores': scores[ids][keep]}


    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """
        Freeze BatchNorm layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def load_state_dict(self, name):
        if name=='FPN50':   
            
            try:
                d = torch.load('weights/pretrained/resnet50-19c8e357.pth')
            except:
                print('Pretrained weights not found')
                return None

            print('Loading pretrained {}...'.format(name))
            dd = self.fpn.state_dict()
            for k in d.keys():
                if not k.startswith('fc'):  # skip fc layers
                    dd[k] = d[k]
         
            for m in self.fpn.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            for m in self.cls_head.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

            for m in self.loc_head.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

            pi = 0.01
            init.constant_(self.cls_head[-1].bias, -math.log((1 - pi) / pi))

            self.fpn.load_state_dict(dd)
            
            print('Loaded pretrained model!')
