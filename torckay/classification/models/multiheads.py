from collections import OrderedDict
import timm
from timm.models.layers import SelectAdaptivePool2d

import torch
import torch.nn as nn

import logging

from torckay.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')

class MultiHeadModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        name="efficientnet_b0",
        pretrained_backbone=None,
        num_head_classes=[1,1],
        train_index=0,
        txt_classnames=None
    ):
        super().__init__()
        self.name = name
        self.train_index = train_index
        self.txt_classnames = txt_classnames
        if txt_classnames is not None:
            self.txt_classnames = self.load_classnames()

        model = timm.create_model(name, pretrained=True)
        self.drop_rate = model.drop_rate
        self.num_features = model.num_features

        if pretrained_backbone is not None:
            state_dict = torch.load(pretrained_backbone, map_location='cpu')
            try:
                ret = model.load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                LOGGER.text(f'[Warning] Ignoring {e}', level=LoggerObserver.WARN)

        # Remove last head, freeze backbone
        self.backbone = nn.Sequential(*(list(model.children())[:-1]))
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.heads = nn.ModuleList()
        for i, num_classes in enumerate(num_head_classes):
            self.append_classifier(model, num_classes, i)
            if train_index != i:
                self.heads[i].requires_grad = False

    def append_classifier(self, model, num_classes, head_index):
        self.heads.add_module(f'{head_index}', 
          nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type='avg')),
                ('norm', model.head.norm),
                ('flatten', nn.Flatten(1)),
                ('drop', nn.Dropout(model.drop_rate)),
                ('fc', nn.Linear(model.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))
        )

    def load_classnames(self):
        self.classnames = []
        with open(self.txt_classnames, 'r') as f:
            groups = f.read().splitlines()

        for group in groups:
            classnames = group.split()
            self.classnames.append(classnames)
            
    def forward_head(self, x, head_index):
        outputs = self.backbone(x)
        outputs = self.heads[head_index](outputs)
        return outputs

    def forward(self, x):
        outputs = self.forward_head(x, self.train_index)
        return outputs

    def get_prediction(self, adict):
        inputs = adict['input']
        head_index = adict['head_index']
        outputs = self.backbone(inputs)
        outputs = self.heads[head_index](outputs)

        probs = torch.max(torch.softmax(outputs, dim=1))
        outputs = torch.argmax(outputs, dim=1)

        probs = probs.cpu().detach().item()
        classid = outputs.cpu().detach().item()
        classname = self.classnames[head_index][outputs]

        return {
            'class': classid,
            'confidence': probs, 
            'name': classname,
        }