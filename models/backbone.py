# Author: Zylo117

import torch
import torchvision
import numpy as np
from torch import nn

from .effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from .effdet.efficientdet import HeadNet

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_model(args, config):
    NUM_CLASSES = len(config.obj_list)

    if config.pretrained_backbone is None:
        load_weights = True
    else:
        load_weights = False

    net = None

    if config.model_name.startswith('efficientdet'):
        compound_coef = config.model_name.split('-')[1]
        assert compound_coef in [f'd{i}' for i in range(8)], f"efficientdet version {i} is not supported"
        net = EfficientDetBackbone(
            num_classes=NUM_CLASSES, 
            compound_coef=compound_coef, 
            load_weights=load_weights, 
            freeze_backbone = getattr(args, 'freeze_backbone', False),
            pretrained_backbone_path=config.pretrained_backbone,
            image_size=config.image_size)
    elif config.model_name.startswith('fasterrcnn'):
        raise NotImplementedError
    return net

class EfficientDetBackbone(nn.Module):
    def __init__(
        self, 
        num_classes=80, 
        compound_coef='d0', 
        load_weights=False, 
        image_size=[512,512], 
        pretrained_backbone_path=None, 
        freeze_backbone=False, 
        **kwargs):

        super(EfficientDetBackbone, self).__init__()
        self.name = f'efficientdet-{compound_coef}'
        config = get_efficientdet_config(f'tf_efficientdet_{compound_coef}')
        config.image_size = image_size
        config.norm_kwargs=dict(eps=.001, momentum=.01)

        net = EfficientDet(
            config, 
            pretrained_backbone=load_weights, 
            freeze_backbone=freeze_backbone,
            pretrained_backbone_path=pretrained_backbone_path)
            
        net.reset_head(num_classes=num_classes)
        net.class_net = HeadNet(config, num_outputs=config.num_classes)

        self.model = DetBenchTrain(net, config)

    def forward(self, inputs, targets):
        return self.model(inputs, targets)

    def detect(self, inputs, img_sizes, img_scales, conf_threshold=0.001):
        outputs = self.model(inputs, inference=True, img_sizes=img_sizes, img_scales=img_scales)
        outputs = outputs.cpu().numpy()
        out = []
        for i, output in enumerate(outputs):
            boxes = output[:, :4]
            labels = output[:, -1]
            scores = output[:,-2]

            if len(boxes) > 0:
                selected = scores >= conf_threshold
                boxes = boxes[selected].astype(np.int32)
                scores = scores[selected]
                labels = labels[selected]
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out
    
