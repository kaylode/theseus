import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .utils import change_box_order
from ensemble_boxes import weighted_boxes_fusion, nms

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

def resize_postprocessing(boxes, current_img_size, ori_img_size):
    """
    Boxes format xyxy or xywh
    """
    new_boxes = boxes.copy()
    new_boxes[:,0] = (boxes[:,0] * ori_img_size[0])/ current_img_size[0]
    new_boxes[:,2] = (boxes[:,2] * ori_img_size[0])/ current_img_size[0]
    new_boxes[:,1] = (boxes[:,1] * ori_img_size[1])/ current_img_size[1]
    new_boxes[:,3] = (boxes[:,3] * ori_img_size[1])/ current_img_size[1]
    # new_boxes[:,[1, 3]] = (boxes[:,[1, 3]] * ori_img_size[1])/ current_img_size[1]
    return new_boxes

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (width, height)
    if isinstance(boxes, torch.Tensor):
        _boxes = boxes.clone()
        _boxes[:, 0].clamp_(0, img_shape[0])  # x1
        _boxes[:, 1].clamp_(0, img_shape[1])  # y1
        _boxes[:, 2].clamp_(0, img_shape[0])  # x2
        _boxes[:, 3].clamp_(0, img_shape[1])  # y2
    else:
        _boxes = boxes.copy()
        _boxes[:, 0] = np.clip(_boxes[:, 0], 0, img_shape[0])  # x1
        _boxes[:, 1] = np.clip(_boxes[:, 1], 0, img_shape[1])  # y1
        _boxes[:, 2] = np.clip(_boxes[:, 2], 0, img_shape[0])  # x2
        _boxes[:, 3] = np.clip(_boxes[:, 3], 0, img_shape[1])  # y2

    return _boxes

def postprocessing(
        preds, 
        current_img_size=None,  # Need to be square
        ori_img_size=None,
        min_iou=0.5, 
        min_conf=0.1,
        mode=None,
        output_format='xywh'):
    """
    Input: bounding boxes in xyxy format
    Output: bounding boxes in xywh format
    """
    boxes, scores, labels = preds['bboxes'], preds['scores'], preds['classes']

    # Clip boxes in image size
    boxes = clip_coords(boxes, current_img_size)

    current_img_size = current_img_size[0] if current_img_size is not None else None
    if len(boxes) != 0:
        if mode is not None:
            boxes, scores, labels = box_fusion(
                [boxes],
                [scores],
                [labels],
                image_size=current_img_size,
                mode=mode,
                iou_threshold=min_iou)

        indexes = np.where(scores > min_conf)[0]
        
        boxes = boxes[indexes]
        scores = scores[indexes]
        labels = labels[indexes]

        # if ori_img_size is not None and current_img_size is not None:
        #     boxes = resize_postprocessing(boxes, current_img_size=current_img_size, ori_img_size=ori_img_size)

        if output_format == 'xywh':
            boxes = change_box_order(boxes, order='xyxy2xywh')


    return {
        'bboxes': boxes, 
        'scores': scores, 
        'classes': labels}

def box_fusion(
    bounding_boxes, 
    confidence_score, 
    labels, 
    mode='wbf', 
    image_size=None,
    weights=None, 
    iou_threshold=0.5):
    """
    bounding boxes: 
        list of boxes of same image [[box1, box2,...],[...]] if ensemble many models
        list of boxes of single image [[box1, box2,...]] if done on one model
    """

    if image_size is not None:
        boxes = [i*1.0/image_size for i in bounding_boxes]
    else:
        boxes = bounding_boxes.copy()

    if mode == 'wbf':
        picked_boxes, picked_score, picked_classes = weighted_boxes_fusion(
            boxes, 
            confidence_score, 
            labels, 
            weights=weights, 
            iou_thr=iou_threshold, 
            conf_type='avg', #[nms|avf]
            skip_box_thr=0.0001)
    elif mode == 'nms':
        picked_boxes, picked_score, picked_classes = nms(
            boxes, 
            confidence_score, 
            labels,
            weights=weights,
            iou_thr=iou_threshold)

    if image_size is not None:
        picked_boxes = picked_boxes*image_size

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)
