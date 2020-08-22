from __future__ import print_function

import os
import sys
import random
import torch
import torch.utils.data as data
from .encoder import DataEncoder


class RetinaNetCollator(object):
    def __init__(self):
        self.encoder = DataEncoder()

    def __call__(self, batch):
        '''Pad images and encode targets.
        Args:
          batch: tensor of images, list of labels, list of boxes.

        Returns:
          images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x['img'] for x in batch]
        boxes = [x['box'] for x in batch]
        labels = [x['label'] for x in batch]

        num_imgs = len(imgs)
        _, h, w = imgs[0].shape
        imgs= torch.stack(imgs, dim=0)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return {
            'imgs': imgs,
            'boxes': torch.stack(loc_targets),
            'labels': torch.stack(cls_targets)}
