import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def filter_area(boxes, confidence_score, labels, min_area=10):
    """
    Boxes in xywh format
    """

    # dimension of bounding boxes
    width = boxes[:, 2]
    height = boxes[:, 3]

    # boxes areas
    areas = width * height

    picked_index = areas >= min_area

    # Picked bounding boxes
    picked_boxes = boxes[picked_index]
    picked_score = confidence_score[picked_index]
    picked_classes = labels[picked_index]

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)




