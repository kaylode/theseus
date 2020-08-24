import numpy as np
import cv2
import torch
import torchvision
from utils.utils import one_hot_embedding, change_box_order, find_jaccard_overlap, box_nms

def meshgrid(x, y, row_major=True):
    """
    Return meshgrid in range x & y
    :param x: (int) first dim range
    :param y: (int) second dim range
    :param row_major: (bool) row major or column major.
    :return: (tensor) meshgrid, sized [x*y, 2]

    Example:
    >> meshgrid(3, 2)
    0 0
    1 0
    2 0
    0 1
    1 1
    2 1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3, 2, row_major=False)
    0 0
    0 1
    0 2
    1 0
    1 1
    1 2
    [torch.FloatTensor of size 6x2]
    """
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)