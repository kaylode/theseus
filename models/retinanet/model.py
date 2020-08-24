import os, sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from .fpn import getFPN
from .utils import one_hot_embedding


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=80, backbone = 'FPN50', pretrained = False):
        super(RetinaNet, self).__init__()
        self.fpn = getFPN(backbone)
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        
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
                d = torch.load('weights/resnet50-19c8e357.pth')
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



class FocalLoss(nn.Module):
    def __init__(self, num_classes=80):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss_alt(self, x, y):
        """
        Focal loss alternative.

        Args:
        :param x: (tensor) sized [N, D]
        :param y: (tensor) sized [N, ].
        :return:
                (tensor) focal loss.
        """
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu().long(), 1 + self.num_classes)  # [N, 81]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t>0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()


    @staticmethod
    def where(cond, x_1, x_2):
        return (cond.float() * x_1) + ((1 - cond.float()) * x_2)

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets):
        """
        Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
        :param loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
        :param loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
        :param cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
        :param cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
            (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        # print(cls_targets)
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N, #anchors]
        num_pos = pos.data.long().sum()
        # print(num_pos, 'num_pos')

        ##########################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ##########################################################
        if num_pos > 0:
            mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N, #anchors, 4]
            masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos, 4]
            masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos, 4]
            # loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
            regression_diff = torch.abs(masked_loc_targets - masked_loc_preds)
            loc_loss = self.where(torch.le(regression_diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2),
                                  regression_diff - 0.5 / 9.0)
            # use mean() here, so the loc_loss dont have to divide num_pos
            # loc_loss = loc_loss.sum()
            loc_loss = loc_loss.mean()
        else:
            num_pos = 1.
            loc_loss = Variable(torch.Tensor([0]).float().cuda())

        ##########################################################
        # cls_loss = FocalLoss(cls_preds, cls_targets)
        ##########################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: {:.3f} | cls_loss: {:.3f}'.format(loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos),
        #       end=' | ')
        # loss = (loc_loss + cls_loss) / num_pos
        loc_loss = loc_loss
        cls_loss = cls_loss / num_pos
        return loc_loss, cls_loss