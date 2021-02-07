import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

STANDARD_COLORS = [
    'LawnGreen', 'LightBlue' , 'Crimson', 'Gold', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

color_list = standard_to_bgr(STANDARD_COLORS)

def one_hot_embedding(labels, num_classes):
    '''
    Embedding labels to one-hot form.
    :param labels: (LongTensor) class labels, sized [N,].
    :param num_classes: (int) number of classes.
    :return: (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


def box_nms(boxes, scores, threshold=0.5):
    """
    Non Maximum Suppression
    Use custom (very slow) or torchvision non-maximum supression on bounding boxes
    
    :param bboxes: (tensor) bounding boxes, size [N, 4]
    :param scores: (tensor) bbox scores, sized [N]
    :return: keep: (tensor) selected box's indices
    """

    # Torchvision NMS:
    keep = torchvision.ops.boxes.nms(boxes, scores,threshold)
    return keep

def box_nms_numpy(bounding_boxes, confidence_score, labels, threshold=0.2, box_format='xyxy'):
    """
    Non Maximum Suppression
    Use custom (very slow) or torchvision non-maximum supression on bounding boxes
    
    :param bboxes: (tensor) bounding boxes, size [N, 4]
    :param scores: (tensor) bbox scores, sized [N]
    :return: keep: (tensor) selected box's indices
    """

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    if box_format == 'xywh':
        end_x += boxes[:, 0]
        end_y += boxes[:, 1]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_classes = []
    
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_classes.append(labels[index])
        
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def change_box_order(boxes, order):
    """
    Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).
    :param boxes: (tensor) or {np.array) bounding boxes, sized [N, 4]
    :param order: (str) ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy']
    :return: (tensor) converted bounding boxes, size [N, 4]
    """

    assert order in ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy']

    # Convert 1-d to a 2-d tensor of boxes, which first dim is 1
    if isinstance(boxes, torch.Tensor):
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)

        if order == 'xyxy2xywh':
            return torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
        elif order ==  'xywh2xyxy':
            return torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], 1)
        elif order == 'xyxy2cxcy':
            return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # c_x, c_y
                            boxes[:, 2:] - boxes[:, :2]], 1)  # w, h
        elif order == 'cxcy2xyxy':
            return torch.cat([boxes[:, :2] - (boxes[:, 2:] *1.0 / 2),  # x_min, y_min
                            boxes[:, :2] + (boxes[:, 2:] *1.0 / 2)], 1)  # x_max, y_max
    else:
        # Numpy
        new_boxes = boxes.copy()
        if order == 'xywh2xyxy':
            new_boxes[:,2] = boxes[:,0] + boxes[:,2]
            new_boxes[:,3] = boxes[:,1] + boxes[:,3]
            return new_boxes
        elif order == 'xyxy2xywh':
            new_boxes[:,2] = boxes[:,2] - boxes[:,0]
            new_boxes[:,3] = boxes[:,3] - boxes[:,1]
            return new_boxes

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2, order='xyxy'):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    The default box order is (xmin, ymin, xmax, ymax).
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    if order == 'xywh':
        set_1 = change_box_order(set_1, 'xywh2xyxy')
        set_2 = change_box_order(set_2, 'xywh2xyxy')

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()

def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)

def postprocessing(outs, imgs, retransforms = None, out_format='xyxy'):
    for item in outs:
        
        boxes_out = item['bboxes']
        if len(boxes_out) == 0:
            continue
        boxes_out_xywh = change_box_order(boxes_out, order = 'xyxy2xywh')
        new_boxes = retransforms(img = imgs, box=boxes_out_xywh)['box']
        if out_format == 'xyxy':
            new_boxes = change_box_order(new_boxes, order = 'xywh2xyxy')
        item['bboxes'] = new_boxes

    return outs

def draw_boxes(img, preds, obj_list):
    bboxes = preds['bboxes']
    labels = preds['classes']
    scores = preds['scores']
    for i, box in enumerate(bboxes):
        x1,y1,w,h = [int(i) for i in box]
        x2 = x1+w
        y2 = y1+h
    
        label = labels[i]
        score = np.round(scores[i], 3)
        color = color_list[label]
        text = '{}: {}'.format(obj_list[label], str(score))
        
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,text,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
    return img

def draw_boxes_v2(img_name, img, boxes, labels, scores, obj_list=None, figsize=(15,15)):
    """
    Visualize an image with its bouding boxes
    """
    fig,ax = plt.subplots(figsize=figsize)

    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze().transpose((1,2,0))
    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch
    for box, label, score in zip(boxes, labels, scores):
        color = STANDARD_COLORS[label]
        x,y,w,h = box
        rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor = color,facecolor='none')
        score = np.round(score, 3)
        if obj_list is not None:
            text = '{}: {}'.format(obj_list[label], str(score))
        else:
            text = '{}: {}'.format(label, str(score))
        plt.text(x, y-3,text, color = color, fontsize=15)
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(img_name,bbox_inches='tight')
    plt.close()

def draw_pred_gt_boxes(image_outname, img, boxes, labels, scores, image_name=None, figsize=(15,15)):
    """
    Visualize an image with its bouding boxes
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    if image_name is not None:
        fig.suptitle(image_name)
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze().transpose((1,2,0))
    # Display the image
    ax1.imshow(img)
    ax2.imshow(img)
    
    ax1.set_title('Prediction')
    ax2.set_title('Ground Truth')

    # Split prediction  and ground truth
    pred_boxes, pred_labels, pred_scores = boxes[0], labels[0], scores
    gt_boxes, gt_labels = boxes[1], labels[1]

    # Plot prediction boxes
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        label = int(label)
        color = STANDARD_COLORS[label]
        x,y,w,h = box
        rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor = color,facecolor='none')
        score = np.round(score, 3)
        text = '{}: {}'.format(label, str(score))
        ax1.text(x, y-3,text, color = color, fontsize=15)
        # Add the patch to the Axes
        ax1.add_patch(rect)

    # Plot ground truth boxes
    for box, label in zip(gt_boxes, gt_labels):
        label = int(label)
        if label <0:
            continue
        color = STANDARD_COLORS[label]
        x,y,w,h = box
        rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor = color,facecolor='none')
        score = np.round(score, 3)
        text = '{}'.format(label)
        ax2.text(x, y-3,text, color = color, fontsize=15)
        # Add the patch to the Axes
        ax2.add_patch(rect)

    plt.axis('off')
    plt.savefig(image_outname,bbox_inches='tight')
    plt.close()

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1



